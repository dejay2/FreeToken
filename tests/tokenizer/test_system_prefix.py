from copy import deepcopy

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager


class Tokenizer:
    name_or_path = "test"
    chat_template = "fixture"

    def __init__(self, mode="normal", pairs=False):
        self.mode, self.pairs = mode, pairs

    def get_chat_template(self, **kwargs):
        return self.chat_template

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize and add_generation_prompt
        assert any(m['role'] == 'user' for m in messages), 'strict template needs a user'
        prefix = '<tools>' + str(kwargs.get('tools', [])) + str(kwargs.get('enable_thinking'))
        rendered = prefix + ''.join('<' + m['role'] + '>' + m['content'] for m in messages)
        if self.mode == 'duplicate':
            rendered += messages[-1]['content']
        if self.mode == 'reorder':
            rendered = messages[-1]['content'] + rendered
        return rendered + '<assistant>'

    def encode(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == 'pt' and not add_special_tokens
        # Pair tokens make the prefix/user boundary sometimes merge into one token.
        ids = [ord(c) for c in text] if not self.pairs else [
            ord(text[i]) * 65536 + (ord(text[i + 1]) if i + 1 < len(text) else 0)
            for i in range(0, len(text), 2)]
        return torch.tensor([ids])


def message(task='hi'):
    return TokenizeMsg(uid=0, sampling_params=SamplingParams(), text=[
        {'role':'system', 'content':'Shared instructions'},
        {'role':'developer', 'content':'Developer rules'},
        {'role':'user', 'content':task}], tools=[{'name':'read'}],
        chat_template_kwargs={'enable_thinking':False}, preserve_system_order=True)


@pytest.mark.parametrize('pairs', [False, True])
def test_system_prefix_is_reusable_across_tasks_and_preserves_inputs(pairs):
    manager = TokenizeManager(Tokenizer(pairs=pairs))
    first, second = message('hi'), message('A different task with a long private tail')
    before = deepcopy(first)
    ids = manager.tokenize([first])[0]
    boundary = manager.system_prefix_tokens(first, ids)
    other_ids = manager.tokenize([second])[0]
    assert boundary == manager.system_prefix_tokens(second, other_ids)
    assert torch.equal(ids[:boundary], other_ids[:boundary])
    visible_prefix = manager.render_prompt(first).split('hi<assistant>')[0]
    assert boundary == len(visible_prefix) // (2 if pairs else 1)
    assert first == before


@pytest.mark.parametrize('mode', ['duplicate', 'reorder'])
def test_ambiguous_templates_refuse_automatic_scope(mode):
    manager = TokenizeManager(Tokenizer(mode=mode))
    msg = message()
    with pytest.raises(ValueError, match='boundary|template'):
        manager.system_prefix_tokens(msg, manager.tokenize([msg])[0])


def test_no_leading_system_does_not_capture_later_system_or_task():
    manager = TokenizeManager(Tokenizer())
    msg = message()
    msg.text = [{'role':'user','content':'private'}, {'role':'system','content':'late rules'}]
    with pytest.raises(ValueError, match='leading system'):
        manager.system_prefix_tokens(msg, manager.tokenize([msg])[0])


def test_template_that_moves_private_text_before_marker_is_rejected():
    manager = TokenizeManager(Tokenizer())
    msg = message()
    original = manager.tokenize([msg])[0]
    manager.render_prompt = lambda probe: ('original prefix ' if probe is msg else 'different prefix ') + probe.text[-1]['content']
    with pytest.raises(ValueError, match='boundary|template'):
        manager.system_prefix_tokens(msg, original)


def test_later_system_hoisting_does_not_accidentally_become_shared():
    manager = TokenizeManager(Tokenizer())
    msg = message()
    msg.text = [msg.text[0], msg.text[-1], {'role':'system','content':'Private later update'}]
    with pytest.raises(ValueError, match='conversation history'):
        manager.system_prefix_tokens(msg, manager.tokenize([msg])[0])
