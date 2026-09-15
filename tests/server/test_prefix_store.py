"""Durable registration behavior: isolation, corruption safety and write bounds."""
import importlib
import json
import os

import pytest


def store_class():
    # A missing implementation is an assertion failure during the initial red run.
    assert importlib.util.find_spec('freetoken.server.prefix_store') is not None
    return importlib.import_module('freetoken.server.prefix_store').PrefixStore


def registration(name='agent'):
    return {'name': name, 'format': 'openai', 'request': {'messages': [{'role': 'system', 'content': 'shared instructions'}]}, 'prefix_scope': 'system', 'ttl_seconds': 300}


def test_saved_registration_survives_restart_without_mutable_aliases(tmp_path):
    cls = store_class()
    path = tmp_path / 'private' / 'model.json'
    store = cls(path)
    source = registration()
    store.put(source)
    source['request']['messages'][0]['content'] = 'changed'
    result = store.registrations()
    result[0]['request']['messages'].clear()
    assert cls(path).registrations() == [registration()]
    assert store.registrations() == [registration()]
    if os.name == 'posix':
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_delete_and_retention_survive_restart(tmp_path):
    cls = store_class()
    path = tmp_path / 'model.json'
    store = cls(path)
    store.put(registration())
    store.configure(1234)
    store.delete('agent')
    store.delete('absent')
    restored = cls(path)
    assert restored.registrations() == []
    assert restored.max_retained_bytes == 1234
    for invalid in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            restored.configure(invalid)
    assert cls(path).max_retained_bytes == 1234


@pytest.mark.parametrize('body', ['{', '[]', '{"version":2,"registrations":{}}', '{"version":1,"registrations":[]}', '{"version":1,"registrations":{"wrong":{"name":"other","format":"openai","request":{}}}}', '{"version":1,"registrations":{},"max_retained_bytes":-1}'])
def test_corrupt_file_is_rejected_and_preserved(tmp_path, body):
    path = tmp_path / 'model.json'
    path.write_text(body)
    with pytest.raises(ValueError):
        store_class()(path)
    assert path.read_text() == body


def test_failed_replace_preserves_memory_and_disk(tmp_path, monkeypatch):
    cls = store_class()
    path = tmp_path / 'model.json'
    store = cls(path)
    store.put(registration())
    original = path.read_bytes()
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError):
        store.put(registration('second'))
    assert path.read_bytes() == original
    assert store.registrations() == [registration()]
    assert list(tmp_path.iterdir()) == [path]


def test_capacity_and_size_limits_preserve_previous_store(tmp_path):
    cls = store_class()
    store = cls(tmp_path / 'model.json')
    for i in range(64):
        store.put(registration(f'agent-{i}'))
    with pytest.raises(ValueError):
        store.put(registration('overflow'))
    huge = registration('agent-0')
    huge['request']['padding'] = 'x' * (4 * 1024 * 1024)
    with pytest.raises(ValueError):
        store.put(huge)
    assert len(cls(store.path).registrations()) == 64
    store.path.write_bytes(b' ' * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        cls(store.path)


def test_total_size_limit_applies_to_writes(tmp_path):
    cls = store_class()
    store = cls(tmp_path / 'model.json')
    for i in range(5):
        item = registration(str(i))
        item['request']['padding'] = 'x' * (3 * 1024 * 1024)
        store.put(item)
    with pytest.raises(ValueError):
        store.put(item | {'name': 'overflow'})
    assert len(cls(store.path).registrations()) == 5


def test_factory_scopes_models_and_can_be_disabled(tmp_path, monkeypatch):
    cls = store_class()
    monkeypatch.setenv('FREETOKEN_PREFIX_STORE_DIR', str(tmp_path / 'stores'))
    first = cls.for_model(str(tmp_path / 'model-a'))
    alias = cls.for_model(str(tmp_path / 'nested' / '..' / 'model-a'))
    other = cls.for_model(str(tmp_path / 'model-b'))
    assert first.path == alias.path
    assert first.path != other.path
    first.put(registration())
    assert other.registrations() == []
    monkeypatch.setenv('FREETOKEN_PREFIX_STORE_DIR', '')
    assert cls.for_model('anything') is None
    monkeypatch.delenv('FREETOKEN_PREFIX_STORE_DIR')
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    assert cls.for_model('anything').path.parent == tmp_path / 'state' / 'freetoken' / 'prefixes'
