import asyncio
from types import SimpleNamespace

from freetoken.server.prefix_api import PrefixRegistrationRequest
from freetoken.server.prefix_store import PrefixStore

BODY = {'name':'saved','format':'openai','prefix_scope':'system','request':{
    'model':'local','messages':[{'role':'system','content':'rules '*100},
                              {'role':'user','content':'hi'}]},'ttl_seconds':300}

class State:
    maintenance_state = 'serving'
    config = SimpleNamespace(model_path='local')
    def __init__(self):
        self.calls=[]
        self.fail_register=False
    async def dispatch_prefix(self, msg, **kwargs):
        self.calls.append(msg)
        if self.fail_register and msg.action=='register':
            return {'status':'invalid','result':{},'error':'too large'}
        return {'status':'ok','result':{'name':msg.name,'state':'ready'},'error':None}


def test_startup_registers_original_template_and_warms_without_generation(tmp_path):
    from freetoken.server.prefix_startup import PrefixStartup
    state=State(); store=PrefixStore(tmp_path/'saved.json'); store.put(BODY); store.configure(12345)
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    asyncio.run(startup.restore())
    assert [m.action for m in state.calls]==['configure','register','warm','get']
    reg=state.calls[1]
    assert reg.prefix_scope=='system' and reg.text[-1]['content']=='hi'
    assert startup.status['state']=='ready'
    assert startup.status['errors']=={}


def test_rejected_restore_keeps_definition_and_reports_error(tmp_path):
    from freetoken.server.prefix_startup import PrefixStartup
    state=State(); state.fail_register=True
    store=PrefixStore(tmp_path/'saved.json'); store.put(BODY)
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    asyncio.run(startup.restore())
    assert [m.action for m in state.calls]==['register']
    assert startup.status['errors']['saved']=='too large'
    assert PrefixStore(store.path).registrations()==[BODY]


def test_failed_backend_registration_is_not_persisted(tmp_path):
    from freetoken.server.prefix_startup import PrefixStartup
    state=State(); state.fail_register=True; store=PrefixStore(tmp_path/'saved.json')
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    response=asyncio.run(startup.register(PrefixRegistrationRequest(**BODY)))
    assert response.status_code==400
    assert store.registrations()==[]


def test_successful_register_survives_restart_and_delete_removes_disk_copy(tmp_path):
    from freetoken.server.prefix_startup import PrefixStartup
    state=State(); store=PrefixStore(tmp_path/'saved.json')
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    assert asyncio.run(startup.register(PrefixRegistrationRequest(**BODY))).status_code==200
    assert len(PrefixStore(store.path).registrations())==1
    assert asyncio.run(startup.delete('saved')).status_code==200
    assert PrefixStore(store.path).registrations()==[]


def test_shutdown_cancels_restore_waiting_for_backend(tmp_path):
    from freetoken.server.prefix_startup import PrefixStartup
    state=State(); state.maintenance_state='loading'
    store=PrefixStore(tmp_path/'saved.json'); store.put(BODY)
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    async def scenario():
        task=asyncio.create_task(startup.restore())
        await asyncio.sleep(0)
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        assert task.cancelled()
    asyncio.run(scenario())
    assert state.calls==[]


def test_transient_write_failure_is_reported_and_next_request_can_retry(tmp_path, monkeypatch):
    from freetoken.server.prefix_startup import PrefixStartup
    state=State(); store=PrefixStore(tmp_path/'saved.json')
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    original=store.put
    def fail(_): raise OSError('disk full')
    monkeypatch.setattr(store, 'put', fail)
    response=asyncio.run(startup.register(PrefixRegistrationRequest(**BODY)))
    assert response.status_code==500 and b'disk full' in response.body
    monkeypatch.setattr(store, 'put', original)
    assert asyncio.run(startup.register(PrefixRegistrationRequest(**BODY))).status_code==200
    assert len(PrefixStore(store.path).registrations())==1


def test_cancelled_writer_holds_lock_until_disk_replace_finishes(tmp_path, monkeypatch):
    from freetoken.server.prefix_startup import PrefixStartup
    import threading
    state=State(); store=PrefixStore(tmp_path/'saved.json')
    startup=PrefixStartup(lambda:state, lambda:{}, store=store)
    entered=threading.Event(); release=threading.Event(); original=store.put
    def slow(recipe):
        entered.set()
        assert release.wait(3)
        original(recipe)
    monkeypatch.setattr(store,'put',slow)
    async def scenario():
        task=asyncio.create_task(startup.register(PrefixRegistrationRequest(**BODY)))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        second=asyncio.create_task(startup.configure(1234))
        try:
            await asyncio.sleep(.05)
            assert [m.action for m in state.calls]==['register']
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await second
    asyncio.run(scenario())
    loaded=PrefixStore(store.path)
    assert len(loaded.registrations())==1 and loaded.max_retained_bytes==1234
