"""Private model-scoped registration recipes; never cache tensors or generated text.

The serving frontend owns this store. Callers serialize mutations in their event
loop; independent serving processes must use different store directories.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


class PrefixStore:
    MAX_ENTRIES = 64
    MAX_BYTES = 16 * 1024 * 1024
    MAX_REGISTRATION_BYTES = 4 * 1024 * 1024

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {'version': 1, 'registrations': {}}
        try:
            with self.path.open('rb') as source:
                raw = source.read(self.MAX_BYTES + 1)
        except FileNotFoundError:
            return
        if len(raw) > self.MAX_BYTES:
            raise ValueError('Prefix registration store exceeds 16 MiB')
        try:
            data = json.loads(raw)
            self._validate(data)
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError(f'Invalid prefix registration store: {exc}') from exc
        self._data = data

    @classmethod
    def for_model(cls, model_path: str | Path) -> PrefixStore | None:
        root = os.environ.get('FREETOKEN_PREFIX_STORE_DIR')
        if root == '':
            return None
        if root is None:
            state = os.environ.get('XDG_STATE_HOME') or str(Path.home() / '.local' / 'state')
            root = str(Path(state) / 'freetoken' / 'prefixes')
        identity = str(Path(model_path).expanduser().resolve())
        key = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        return cls(Path(root).expanduser() / f'{key}.json')

    @staticmethod
    def _encode(data: Any) -> bytes:
        return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')

    @classmethod
    def _validate(cls, data: Any) -> None:
        if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
            raise ValueError('expected schema version 1')
        if set(data) - {'version', 'registrations', 'max_retained_bytes'}:
            raise ValueError('unknown store fields')
        entries = data.get('registrations')
        if not isinstance(entries, dict) or len(entries) > cls.MAX_ENTRIES:
            raise ValueError('expected at most 64 named registrations')
        if 'max_retained_bytes' in data:
            value = data['max_retained_bytes']
            if type(value) is not int or value < 0:
                raise ValueError('max_retained_bytes must be a non-negative integer')
        for name, entry in entries.items():
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', name):
                raise ValueError('invalid registration name')
            if not isinstance(entry, dict) or entry.get('name') != name:
                raise ValueError('registration name does not match its key')
            if entry.get('format') not in {'openai', 'anthropic'} or not isinstance(entry.get('request'), dict):
                raise ValueError('registration requires format and request object')
            if set(entry) - {'name', 'format', 'request', 'prefix_tokens', 'prefix_scope', 'ttl_seconds'}:
                raise ValueError('unknown registration fields')
            scope, tokens = entry.get('prefix_scope'), entry.get('prefix_tokens')
            if scope not in (None, 'system') or (scope is not None and tokens is not None):
                raise ValueError('invalid prefix scope')
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError('invalid prefix token count')
            ttl = entry.get('ttl_seconds', 300)
            if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 <= ttl <= 86400:
                raise ValueError('invalid prefix retention TTL')
            if len(cls._encode(entry)) > cls.MAX_REGISTRATION_BYTES:
                raise ValueError('Prefix registration exceeds 4 MiB')
        if len(cls._encode(data)) > cls.MAX_BYTES:
            raise ValueError('Prefix registration store exceeds 16 MiB')

    def registrations(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._data['registrations'].values()))

    @property
    def max_retained_bytes(self) -> int | None:
        return self._data.get('max_retained_bytes')

    def put(self, registration: dict[str, Any]) -> None:
        if not isinstance(registration, dict) or not isinstance(registration.get('name'), str):
            raise ValueError('registration requires a name')
        data = deepcopy(self._data)
        data['registrations'][registration['name']] = deepcopy(registration)
        self._save(data)

    def delete(self, name: str) -> None:
        data = deepcopy(self._data)
        data['registrations'].pop(name, None)
        self._save(data)

    def configure(self, max_retained_bytes: int) -> None:
        data = deepcopy(self._data)
        data['max_retained_bytes'] = max_retained_bytes
        self._save(data)

    def _save(self, data: dict[str, Any]) -> None:
        self._validate(data)
        payload = self._encode(data)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.prefix-', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        self._data = data
