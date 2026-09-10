"""Exercise RAM reuse with Anthropic tools, reasoning and inline system updates.

Uses only synthetic archive facts. Two initial fills are followed by concurrent
continuations and serial revisits. Run against an idle validation server with
RAM parking/cache reporting enabled. Before serial revisits, an idle-only cache
rebuild keeps the same state-slot count but clears GPU prefixes, forcing RAM
reloads without a server restart. Unrelated traffic can evict these families
or overwrite the global restore diagnostic used by the serial assertions.

--check-prefix performs only local rendering/tokenization, without generation.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from freetoken.message import TokenizeMsg
from freetoken.server.anthropic_api import convert_anthropic_to_genspec
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.tokenizer.tokenize import TokenizeManager
from freetoken.utils import load_tokenizer

from kv_long_chat_live import http_json, model_id


def encode(manager, request):
    spec = convert_anthropic_to_genspec(AnthropicMessagesRequest.model_validate(request), {})
    return manager.tokenize([TokenizeMsg(
        uid=0, text=spec.messages, sampling_params=spec.sampling_params,
        tools=spec.template_tools, chat_template_kwargs=spec.chat_template_kwargs,
    )])[0].tolist()


def archive(manager, model, family, target, trial):
    values = dict(zip(('first', 'middle', 'last'), (
        ('maple7319', 'copper2048', 'violet9631') if family == 'A'
        else ('amber5826', 'silver1073', 'willow6402')
    )))
    request = {
        'model': model, 'max_tokens': 1024, 'temperature': 0, 'stream': True,
        'thinking': {'type': 'enabled', 'budget_tokens': 1024},
        'system': f'You are reading independent archive {family}, trial {trial}. Remember its three facts. Call report_facts with those values when asked.',
        'tools': [{'name': 'report_facts', 'description': 'Report the three archive facts exactly.',
                   'input_schema': {'type': 'object', 'properties': {
                       key: {'type': 'string'} for key in values
                   }, 'required': list(values), 'additionalProperties': False}}],
        'messages': [],
    }
    filler = f'Archive {family}: ordinary gray stones were counted beside the river.\n'
    per_copy = len(manager.tokenizer.encode(filler, add_special_tokens=False))
    copies = target // per_copy
    for _ in range(64):
        text = (f'FIRST={values["first"]}.\n' + filler * (copies // 2)
                + f'\nMIDDLE={values["middle"]}.\n' + filler * (copies - copies // 2)
                + f'\nLAST={values["last"]}.\nCall report_facts with FIRST, MIDDLE and LAST from this archive.')
        request['messages'] = [
            {'role': 'user', 'content': text},
            {'role': 'system', 'content': '<total_tokens>90000 tokens left</total_tokens>'},
        ]
        length = len(encode(manager, request))
        if target <= length <= target + 128:
            return request, values
        copies += max(1, (target - length) // per_copy) if length < target else min(-1, (target - length) // per_copy)
    raise AssertionError('could not size synthetic archive')


def continuation(request, content, turn):
    request = copy.deepcopy(request)
    calls = [block for block in content if block['type'] == 'tool_use']
    request['messages'].extend([
        {'role': 'assistant', 'content': content},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': call['id'], 'content': 'Recorded.'}
            for call in calls
        ]},
        {'role': 'system', 'content': f'<total_tokens>{90000 - turn * 1000} tokens left</total_tokens>'},
        {'role': 'user', 'content': 'Call report_facts again with the same three archive values.'},
    ])
    return request


def send(url, request, expected):
    started = time.monotonic()
    first_token = None
    blocks, partial, usage = {}, {}, {}
    stop_reason = None
    wire = Request(url + '/v1/messages', data=json.dumps(request).encode(),
                   headers={'Content-Type': 'application/json'})
    with urlopen(wire, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            event = json.loads(raw[6:])
            kind = event['type']
            if kind == 'error':
                raise AssertionError(event)
            if kind == 'message_start':
                usage.update(event['message']['usage'])
            elif kind == 'content_block_start':
                blocks[event['index']] = event['content_block']
            elif kind == 'content_block_delta':
                if first_token is None:
                    first_token = time.monotonic() - started
                index, delta = event['index'], event['delta']
                if delta['type'] == 'input_json_delta':
                    partial[index] = partial.get(index, '') + delta['partial_json']
                elif delta['type'] == 'thinking_delta':
                    blocks[index]['thinking'] = blocks[index].get('thinking', '') + delta['thinking']
                elif delta['type'] == 'text_delta':
                    blocks[index]['text'] = blocks[index].get('text', '') + delta['text']
            elif kind == 'content_block_stop' and event['index'] in partial:
                index = event['index']
                blocks[index]['input'] = json.loads(partial[index])
            elif kind == 'message_delta':
                usage.update(event.get('usage', {}))
                stop_reason = event['delta'].get('stop_reason')
    content = [blocks[index] for index in sorted(blocks)]
    calls = [block for block in content if block['type'] == 'tool_use']
    assert stop_reason == 'tool_use' and len(calls) == 1, {'stop': stop_reason, 'blocks': content}
    assert calls[0]['name'] == 'report_facts' and calls[0]['input'] == expected, calls
    return content, {'usage': usage, 'ttft_s': first_token, 'seconds': time.monotonic() - started,
                     'thinking_chars': sum(len(b.get('thinking', '')) for b in content),
                     'reported_facts': calls[0]['input']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:2020')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--tokens', type=int, default=100_000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-prefix', action='store_true')
    args = parser.parse_args()
    manager = TokenizeManager(load_tokenizer(args.model_path))
    url = args.base_url.rstrip('/')
    model = 'local-prefix-check' if args.check_prefix else model_id(url)
    trial = uuid.uuid4().hex[:8]
    requests, facts = {}, {}
    for index, family in enumerate('AB'):
        requests[family], facts[family] = archive(manager, model, family, args.tokens + index * 1024, trial)
    if args.check_prefix:
        for family in 'AB':
            response = [{'type': 'thinking', 'thinking': 'Read all three facts.'},
                        {'type': 'tool_use', 'id': 'call1', 'name': 'report_facts', 'input': facts[family]}]
            before = encode(manager, requests[family])
            after = encode(manager, continuation(requests[family], response, 1))
            checkpoint = (len(before) - 1) // 64 * 64
            assert before[:checkpoint] == after[:checkpoint], 'appended system update changed old prefix'
        print(json.dumps({'event': 'prefix_pass', 'families': 2, 'tokens': args.tokens}), flush=True)
        return

    def get(path):
        return http_json(url + path, timeout=20)[1]

    initial = get('/v1/cache/status')
    assert initial['parking']['mode'] == 'ram', 'enable RAM parking before running'
    instance = get('/health')['instance_id']
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(result):
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps({'complete': False, 'instance': instance, 'results': results}, indent=2) + '\n')

    def clear_gpu_prefixes():
        # A normal GPU hit is valid behavior, so explicitly park/reset its radix
        # before demanding a host transfer. if_idle refuses active requests;
        # neither this test nor the endpoint aborts them to obtain the safe point.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            current = get('/v1/cache/status')
            if current['state'] != 'serving' or get('/v1/stats')['requests']['active']:
                time.sleep(0.5)
                continue
            slots = current['geometry']['num_mamba_slots']
            body = {'mode': 'if_idle', 'num_mamba_slots': slots}
            wire = Request(url + '/v1/cache/rebuild', data=json.dumps(body).encode(),
                           headers={'Content-Type': 'application/json'})
            try:
                with urlopen(wire, timeout=360) as response:
                    rebuilt = json.load(response)
            except HTTPError as exc:
                detail = json.loads(exc.read())
                if detail.get('status') == 'busy':
                    time.sleep(0.5)
                    continue
                raise
            assert rebuilt['status'] == 'ok' and rebuilt['num_mamba_slots'] == slots, rebuilt
            assert get('/health')['instance_id'] == instance, 'server restarted'
            print(json.dumps({'event': 'gpu_prefixes_cleared', 'state_slots': slots}), flush=True)
            return
        raise TimeoutError('validation server did not become idle for GPU-prefix displacement')

    def run_one(family, turn):
        local_tokens = len(encode(manager, requests[family]))
        counted = http_json(url + '/v1/messages/count_tokens', requests[family], timeout=60)[1]
        assert counted['input_tokens'] == local_tokens, 'frontend and local tokenizer disagree'
        before = get('/v1/cache/status')['parking']
        content, result = send(url, requests[family], facts[family])
        after = get('/v1/cache/status')['parking']
        result.update(family=family, turn=turn, ram_hits_delta=after['hits'] - before['hits'],
                      restore=after['last_restore_breakdown_ms'])
        assert get('/health')['instance_id'] == instance, 'server restarted'
        assert not after['disabled'] and after['last_error'] is None, after
        cached = result['usage'].get('cache_read_input_tokens') or 0
        # Anthropic input_tokens is uncached input; add cached to recover total.
        total = result['usage']['input_tokens'] + cached
        result.update(cached_tokens=cached, total_prompt_tokens=total)
        assert total == local_tokens, 'generation and token counting disagree'
        assert total >= args.tokens, result
        if turn == 0:
            assert result['thinking_chars'] > 0, 'initial response omitted reasoning; replay would not be exercised'
        if turn:
            assert cached >= args.tokens - 64, 'old archive was reprocessed'
            assert total - cached <= 4096, 'unexpectedly large new prefill'
        if turn == 2:
            transfer = result['restore']
            assert transfer.get('sequence', 0) > before['last_restore_breakdown_ms'].get('sequence', 0), 'no completed RAM restore'
            assert transfer['page_offset'] == 0, 'prefix remained partly on GPU'
            units = initial['geometry']['unit_bytes']
            assert transfer['kv_bytes'] == cached * units['kv_per_token'], 'restore cannot be attributed to this request'
            assert transfer['state_bytes'] == units['mamba_per_slot'], 'incomplete recurrent state'
        requests[family] = continuation(requests[family], content, turn + 1)
        return result

    for family in 'AB':
        record(run_one(family, 0))
        time.sleep(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_one, family, 1) for family in 'AB']
        for future in futures:
            record(future.result())
    time.sleep(2)
    clear_gpu_prefixes()
    for family in 'AB':
        record(run_one(family, 2))
        time.sleep(2)
    args.output.write_text(json.dumps({'complete': True, 'instance': instance, 'results': results}, indent=2) + '\n')
    print(json.dumps({'event': 'agent_ram_pass', 'requests': len(results), 'instance': instance}), flush=True)


if __name__ == '__main__':
    main()
