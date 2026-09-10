"""Real Jinja/encoding regressions for appended Anthropic system updates.

The small fixture retains the relevant structure of the serving Qwen3.8
template: a leading tools/system block, strict late-system guard, reasoning,
and tool turns. Full-checkpoint rendering is also checked in the live benchmark.
"""
from __future__ import annotations

import copy
import json

import pytest
from jinja2 import TemplateError
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.server.anthropic_api import convert_anthropic_to_genspec
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.openai_api import chat_request_to_genspec
from freetoken.server.responses_api import ResponsesRequest, convert_responses_to_genspec
from freetoken.tokenizer.tokenize import TokenizeManager


QWEN_TEMPLATE = r"""
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}{{- content }}
    {%- elif content is iterable %}
        {%- for item in content %}
            {%- if item.type == 'image' %}
                {%- if is_system_content %}{{- raise_exception('System message cannot contain images.') }}{%- endif %}
                {{- '<|image_pad|>' }}
            {%- else %}{{- item.text }}{%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro %}
{%- if tools or messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' }}
    {%- if tools %}{{- tools|tojson + '\n' }}{%- endif %}
    {%- if messages[0].role == 'system' %}{{- render_content(messages[0].content, false, true)|trim }}{%- endif %}
    {{- '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n<think>\n' + message.get('reasoning_content', '') + '\n</think>\n\n' + content }}
        {%- if message.tool_calls %}{{- '\n<tool_call>' + message.tool_calls|tojson + '</tool_call>' }}{%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == 'user' or message.role == 'tool' %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- else %}{{- raise_exception('Unsupported role') }}{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n' }}{%- endif %}
"""


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["archive system user assistant tool tokens left remember violet maple reasoning"],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=QWEN_TEMPLATE)


def as_msg(raw):
    spec = convert_anthropic_to_genspec(AnthropicMessagesRequest.model_validate(raw), {})
    return TokenizeMsg(
        uid=1, text=spec.messages, sampling_params=spec.sampling_params,
        tools=spec.template_tools, chat_template_kwargs=spec.chat_template_kwargs,
        preserve_system_order=spec.preserve_system_order,
    )


def agent_request():
    return {
        "model": "qwen", "max_tokens": 64,
        "system": "Remember the archive facts.",
        "tools": [{"name": "lookup", "description": "Look up a record.",
                   "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": "Archive: maple and violet. " * 256},
            {"role": "system", "content": "<total_tokens>90000 tokens left</total_tokens>"},
        ],
    }


def test_appended_counter_keeps_old_prompt_checkpoint_prefix(tokenizer):
    """Hoisting a later budget update must break this full-token-prefix check."""
    first = agent_request()
    second = copy.deepcopy(first)
    second["messages"].extend([
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Find the first archive entry."},
            {"type": "tool_use", "id": "call1", "name": "lookup", "input": {"key": "maple"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call1", "content": "violet"}]},
        {"role": "system", "content": "<total_tokens>89000 tokens left</total_tokens>"},
    ])
    manager = TokenizeManager(tokenizer)
    first_msg, second_msg = as_msg(first), as_msg(second)
    old, new = manager.tokenize([first_msg, second_msg])
    checkpoint = (len(old) - 1) // 64 * 64
    assert checkpoint > 1000
    assert new[:checkpoint].tolist() == old[:checkpoint].tolist()
    prompt = manager.render_prompt(second_msg)
    old_counter = '<|im_start|>system\n<total_tokens>90000 tokens left</total_tokens><|im_end|>'
    new_counter = '<|im_start|>system\n<total_tokens>89000 tokens left</total_tokens><|im_end|>'
    assert prompt.index('Archive:') < prompt.index(old_counter) < prompt.index('Find the first')
    assert prompt.index('</tool_call>') < prompt.index(new_counter)
    assert '<|im_start|>tool\nviolet<|im_end|>' in prompt
    assert first_msg.text == as_msg(first).text  # render must not hoist in place


def test_supported_template_keeps_system_updates_after_history(tokenizer):
    req = agent_request()
    manager = TokenizeManager(tokenizer)
    prompt = manager.render_prompt(as_msg(req))
    assert prompt.count('<|im_start|>system\n') == 2
    assert prompt.index('Archive:') < prompt.index('90000 tokens left')


@pytest.mark.parametrize('with_tools', [False, True])
def test_normal_requests_keep_exact_template_bytes(tokenizer, with_tools):
    raw = agent_request()
    raw['messages'].pop()
    if not with_tools:
        raw.pop('tools')
    msg = as_msg(raw)
    original = tokenizer.apply_chat_template(
        msg.text, tools=msg.tools, tokenize=False, add_generation_prompt=True,
        **msg.chat_template_kwargs,
    )
    assert TokenizeManager(tokenizer).render_prompt(msg) == original
    assert tokenizer.chat_template == QWEN_TEMPLATE


def test_unknown_template_keeps_existing_system_hoisting(tokenizer):
    tokenizer.chat_template = """{%- for m in messages %}
{%- if m.role == 'system' and not loop.first %}{{- raise_exception('Only one leading system supported') }}{%- endif %}
{{- '[' + m.role + ']' + m.content }}{%- endfor %}"""
    msg = as_msg(agent_request())
    original_messages = copy.deepcopy(msg.text)
    prompt = TokenizeManager(tokenizer).render_prompt(msg)
    assert prompt.startswith('[system]Remember the archive facts.\n\n<total_tokens>90000 tokens left</total_tokens>[user]Archive:')
    assert msg.text == original_messages


def test_unrelated_template_validation_is_not_bypassed(tokenizer):
    # An accepted system layout must not hide unrelated invalid roles.
    msg = as_msg(agent_request())
    msg.text.append({'role': 'invalid', 'content': 'bad'})
    with pytest.raises(TemplateError, match='Unsupported role'):
        TokenizeManager(tokenizer).render_prompt(msg)


def test_late_system_images_keep_system_validation(tokenizer):
    # Native Anthropic currently drops images; check the renderer's own contract
    # so relaxing the position guard cannot also relax the media restriction.
    msg = as_msg(agent_request())
    msg.text[-1]['content'] = [{'type': 'image', 'image_url': 'example'}]
    with pytest.raises(TemplateError, match='System message cannot contain images'):
        TokenizeManager(tokenizer).render_prompt(msg)


@pytest.mark.parametrize('protocol', ['chat_completions', 'responses'])
def test_client_template_kwarg_cannot_enable_system_order_adapter(tokenizer, protocol):
    template_kwargs = {'_freetoken_preserve_system_order': True}
    if protocol == 'chat_completions':
        spec = chat_request_to_genspec(ChatCompletionRequest.model_validate({
            'model': 'qwen',
            'messages': [
                {'role': 'user', 'content': 'hello'},
                {'role': 'system', 'content': 'later'},
            ],
            'chat_template_kwargs': template_kwargs,
        }), {})
    else:
        spec = convert_responses_to_genspec(ResponsesRequest.model_validate({
            'model': 'qwen', 'input': 'hello',
            'chat_template_kwargs': template_kwargs,
        }), {})
    assert spec.preserve_system_order is False
    assert spec.chat_template_kwargs == template_kwargs
    msg = TokenizeMsg(
        uid=1,
        sampling_params=SamplingParams(),
        text=[
            {'role': 'user', 'content': 'hello'},
            {'role': 'system', 'content': 'later'},
        ],
        chat_template_kwargs=spec.chat_template_kwargs,
        preserve_system_order=spec.preserve_system_order,
    )
    with pytest.raises(TemplateError, match='System message must be at the beginning'):
        TokenizeManager(tokenizer).render_prompt(msg)


def test_internal_signal_without_late_system_does_not_trigger_fallback(tokenizer):
    tokenizer.chat_template = '{{ messages|tojson }}'
    msg = TokenizeMsg(
        uid=1,
        sampling_params=SamplingParams(),
        text=[
            {'role': 'system', 'content': 'rules', 'name': 'retained'},
            {'role': 'user', 'content': 'hello'},
        ],
        preserve_system_order=True,
    )
    assert json.loads(TokenizeManager(tokenizer).render_prompt(msg)) == msg.text


def test_tool_specific_template_is_selected_before_adapting(tokenizer):
    tokenizer.chat_template = {'default': '{{ messages|tojson }}', 'tool_use': QWEN_TEMPLATE}
    prompt = TokenizeManager(tokenizer).render_prompt(as_msg(agent_request()))
    assert prompt.count('<|im_start|>system\n') == 2
    assert prompt.index('Archive:') < prompt.index('90000 tokens left')
    assert tokenizer.chat_template['tool_use'] == QWEN_TEMPLATE


def test_custom_template_override_keeps_legacy_fallback_and_client_kwargs(tokenizer):
    msg = as_msg(agent_request())
    msg.chat_template_kwargs.update({
        '_freetoken_preserve_system_order': 'visible-client-data',
        'chat_template': "{{ [_freetoken_preserve_system_order, messages]|tojson }}",
    })
    visible, rendered = json.loads(TokenizeManager(tokenizer).render_prompt(msg))
    assert visible == 'visible-client-data'
    assert [m['role'] for m in rendered] == ['system', 'user']
    assert rendered[0]['content'] == 'Remember the archive facts.\n\n<total_tokens>90000 tokens left</total_tokens>'


def test_custom_encoder_retains_legacy_hoisting(tokenizer, tmp_path):
    encoding = tmp_path / 'encoding'
    encoding.mkdir()
    (encoding / 'encoding_dsv4.py').write_text('''
import json
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    return json.dumps(messages)
''')
    tokenizer.chat_template = None
    tokenizer.name_or_path = str(tmp_path)
    raw = agent_request()
    raw.pop('tools')
    msg = as_msg(raw)
    before = copy.deepcopy(msg)
    rendered = json.loads(TokenizeManager(tokenizer).render_prompt(msg))
    assert rendered == [
        {'role': 'system', 'content': 'Remember the archive facts.\n\n<total_tokens>90000 tokens left</total_tokens>'},
        {'role': 'user', 'content': 'Archive: maple and violet. ' * 256},
    ]
    assert msg == before
