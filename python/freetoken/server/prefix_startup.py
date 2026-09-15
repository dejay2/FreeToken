"""Frontend-owned saved registrations and generation-free startup warming."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from freetoken.message import PrefixCacheMsg
from .prefix_store import PrefixStore

logger = logging.getLogger(__name__)
_UNSET = object()


class PrefixStartup:
    def __init__(self, get_state, get_sampling, *, store=_UNSET):
        self.get_state, self.get_sampling = get_state, get_sampling
        self._store = store
        self._lock = asyncio.Lock()
        self.status = {'state': 'pending', 'errors': {}}
        self.store_error = None

    @property
    def store(self):
        if self._store is _UNSET:
            state = self.get_state()
            model = getattr(getattr(state, 'config', None), 'model_path', None)
            if not model:
                return None  # lightweight route tests / no model initialized yet
            try:
                self._store = PrefixStore.for_model(model)
            except (OSError, ValueError) as exc:
                self.store_error = str(exc)
                self._store = None
                logger.error('Cannot load saved prompt registrations: %s', exc)
        return self._store

    def metadata(self):
        store = self.store
        return {'enabled': store is not None, 'saved_names': [r['name'] for r in store.registrations()] if store else [],
                'startup': self.status, 'error': self.store_error}

    async def _send(self, action, **kwargs):
        from .prefix_api import _dispatch
        return await _dispatch(self.get_state(), PrefixCacheMsg(
            request_id=str(uuid.uuid4()), action=action, **kwargs))

    async def _register(self, req):
        from .prefix_api import _registration_spec, _response
        try:
            spec = _registration_spec(req, self.get_state(), self.get_sampling())
        except ValueError as exc:
            return _response({'status':'invalid', 'result':{}, 'error':str(exc)})
        return await self._send('register', name=req.name, text=spec.messages,
            tools=spec.template_tools, chat_template_kwargs=spec.chat_template_kwargs,
            preserve_system_order=spec.preserve_system_order, prefix_tokens=req.prefix_tokens,
            prefix_scope=req.prefix_scope, ttl_seconds=req.ttl_seconds)

    @staticmethod
    def _body(response):
        return json.loads(response.body)

    def _failure(self, exc):
        from .prefix_api import _response
        return _response({'status':'failed', 'result':{},
            'error':f'Could not save prompt settings: {exc}'})

    async def _persist(self, operation, value):
        # A disconnected HTTP client must not release the mutation lock while
        # its disk writer is still replacing the store behind another request.
        task = asyncio.create_task(asyncio.to_thread(operation, value))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def register(self, req):
        async with self._lock:
            store = self.store
            if self.store_error:
                return self._failure(self.store_error)
            # Validate storage bounds before mutating the live registry.
            recipe = req.model_dump(exclude_none=True)
            if store:
                try:
                    data = {'version':1, 'registrations':{r['name']:r for r in store.registrations()}}
                    data['registrations'][req.name] = recipe
                    store._validate(data)
                except (ValueError, TypeError) as exc:
                    from .prefix_api import _response
                    return _response({'status':'invalid', 'result':{}, 'error':str(exc)})
            response = await self._register(req)
            if store and response.status_code == 200:
                try:
                    await self._persist(store.put, recipe)
                except (OSError, ValueError) as exc:
                    return self._failure(exc)
            return response

    async def delete(self, name):
        async with self._lock:
            store = self.store
            response = await self._send('delete', name=name)
            if store and response.status_code in (200, 404):
                try:
                    await self._persist(store.delete, name)
                    self.status['errors'].pop(name, None)
                except (OSError, ValueError) as exc:
                    return self._failure(exc)
            return response

    async def configure(self, budget):
        async with self._lock:
            store = self.store
            if self.store_error:
                return self._failure(self.store_error)
            response = await self._send('configure', max_retained_bytes=budget)
            if store and response.status_code == 200:
                try:
                    await self._persist(store.configure, budget)
                except (OSError, ValueError) as exc:
                    return self._failure(exc)
            return response

    async def restore(self):
        from .prefix_api import PrefixRegistrationRequest
        self.status['state'] = 'waiting'
        while True:
            state = self.get_state()
            phase = getattr(state, 'maintenance_state', 'loading')
            if phase == 'serving':
                break
            if phase in ('failed', 'draining'):
                self.status['state'] = 'unavailable'
                return
            await asyncio.sleep(0.2)
        store = self.store
        if store is None:
            self.status['state'] = 'error' if self.store_error else 'disabled'
            return
        self.status['state'] = 'warming'
        async with self._lock:
            if store.max_retained_bytes is not None:
                response = await self._send('configure', max_retained_bytes=store.max_retained_bytes)
                if response.status_code != 200:
                    self.status['errors']['settings'] = self._body(response).get('error')
        for recipe in store.registrations():
            name = recipe['name']
            try:
                async with self._lock:
                    # A user can delete/replace a saved recipe while startup warms others.
                    if recipe not in store.registrations():
                        continue
                    response = await self._register(PrefixRegistrationRequest(**recipe))
                    if response.status_code != 200:
                        raise ValueError(self._body(response).get('error') or 'registration failed')
                    response = await self._send('warm', name=name)
                    if response.status_code not in (200, 202):
                        raise ValueError(self._body(response).get('error') or 'warming failed')
                deadline = time.monotonic() + 300
                while True:
                    response = await self._send('get', name=name)
                    body = self._body(response)
                    result = body.get('result', {})
                    if response.status_code != 200 or result.get('error'):
                        raise ValueError(result.get('error') or body.get('error') or 'warming failed')
                    if result.get('state') == 'ready':
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('startup warming timed out')
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status['errors'][name] = str(exc)
                logger.warning('Saved prompt %s could not be warmed: %s', name, exc)
        self.status['state'] = 'error' if self.status['errors'] else 'ready'
