# Reusable local agent prefixes

The local server can prepare an exact shared prompt once, then let multiple agents reference
the same attention pages. Each admitted agent still gets its own mutable recurrent state and
private continuation pages. Reusing a checkpoint avoids the shared prefix's prefill computation;
it does not remove those tokens from the context window or make later decoding free.

Initial support is text QSA/GDN models, hybrid radix caching and `TP=1`. The control API returns
`unsupported` for other engines. Ordinary generation keeps its existing behavior. This feature
uses one local FreeToken server, its GPU, local RAM and local SSD; multi-server sharing is not
part of the design.

## Try it from the settings page

Open `http://127.0.0.1:2031` and choose **Prompt cache**. With the model server ready:

1. Choose **Load test example**, then **Register & test**. The default sends two
   identical requests, each limited to 64 output tokens, and reports their answers,
   elapsed times and cached input tokens. Select 1, 2, 4 or 8 agents to change the test;
   requests above the server's concurrency limit queue normally.
2. Use **Register prompt** to register without generating, **Warm** to prepare a
   registered cache, and **Delete** to remove an alias. Watch shared token counts,
   preparations, followers, restores and GPU/RAM/SSD residency in the list.
3. Paste your shared instructions and a test task, or switch to **Full request JSON**
   for the exact OpenAI/Anthropic messages, tools and template settings your agents use.
   The simple editor uses the running model with thinking disabled. Set **Shared prefix
   tokens** inside the common instructions, before variable text. Blank requests the
   entire rendered request for identical replays; it does not automatically find the
   system-prompt boundary.

The per-prompt GPU preference duration and shared retention budget apply immediately;
they do not require Save or restart. Both are best-effort preferences, not reserved
VRAM. Names and live retention changes are lost on a model-server restart. Prompt drafts
stay in the current page during refresh errors, but are not saved across page reloads.
The test displays model text and tool calls; it does not execute tools.

**Recent requests** lists up to 50 accepted text requests from `/v1/chat/completions`
and `/v1/messages` from the past hour. Choose **Use prompt** to load the original JSON,
including tools and template options, into the editor. It resets the shared-token field
to whole-request caching; set an explicit boundary before variable task text when sharing
only the common instructions. Selection itself does not register or generate anything.

Collection begins when the updated model server starts. Snapshots stay in local RAM,
with a 16 MiB payload budget and a 4 MiB limit per request; older entries are evicted,
and oversized requests are skipped. Private and image requests are excluded. The list
contains request previews and loads full content only when selected. **Pause collection**
stops new snapshots, and **Clear list** removes existing snapshots without deleting named
caches. Pause lasts for this model-server process; restarting clears the list and resumes
collection. Requests sent before this feature was installed cannot be recovered here.

The management API exposes `GET/DELETE /v1/cache/recent`, `GET /v1/cache/recent/{id}` and
`PUT /v1/cache/recent/settings` with `{"enabled": false}` to pause. These use the same
local access model as the existing cache controls. They are available through the settings
helper under `/api/prompt-cache/recent` as well.

The current pool and parking mode are shown under **RAM, SSD and larger prompts**.
Parking settings and **Smallest KV memory** remain under **Model & chats** and require
a model-server restart when changed. Registration itself cannot grow a dynamic KV pool.
Deleting an alias does not flush pages held by other aliases or active requests.

The settings helper forwards only these cache controls, model/status reads and bounded
test generation to its local server. Updating this UI needs a settings-helper restart;
it does not require reloading an already compatible model server.

## Register the exact common prefix

The registration request accepts an ordinary OpenAI or Anthropic payload. The server renders
and tokenizes it through the same path as generation. `prefix_tokens` selects a prefix of that
rendered request and is rounded down to a whole cache page. Put the cut before variable task
text. Omitting the cut requests the entire rendered prompt, including chat delimiters and the
generation header; that is suitable only when later requests really share those tokens. The
cached boundary always leaves at least one prompt token for normal generation. If the entire
rendered prompt is page aligned, its final page stays private. A prompt too short to leave
both a whole cached page and a generation tail is rejected. `requested_tokens` reports the
original count for each alias; `prefix_tokens` reports its actual shared boundary.

This example creates a registration file from two otherwise identical requests with different
tasks. Run it in the FreeToken Python environment, passing the served tokenizer's exact local
model path. It uses the same converter and tokenizer as the server:

```bash
PYTHONPATH=python python - /path/to/served/model > agent-prefix.json <<'PY'
import copy, json, sys
from freetoken.message import TokenizeMsg
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.openai_api import chat_request_to_genspec
from freetoken.tokenizer.tokenize import TokenizeManager
from freetoken.utils import load_tokenizer

manager = TokenizeManager(load_tokenizer(sys.argv[1]))
request = {
    "model": "local", "max_tokens": 32,
    "messages": [
        {"role": "system", "content": (
            "You are a coding agent. Inspect the files, make the requested change, "
            "and verify it. Keep the user's existing work intact.\n"
        ) * 20},
        {"role": "user", "content": "Inspect the parser."},
    ],
}
def encode(body):
    spec = chat_request_to_genspec(ChatCompletionRequest.model_validate(body), {})
    return manager.tokenize([TokenizeMsg(
        uid=0, text=spec.messages, sampling_params=spec.sampling_params,
        tools=spec.template_tools, chat_template_kwargs=spec.chat_template_kwargs,
        preserve_system_order=spec.preserve_system_order,
    )])[0]
other = copy.deepcopy(request)
other["messages"][-1]["content"] = "Review the tests."
a, b = encode(request), encode(other)
n = min(len(a), len(b))
different = (a[:n] != b[:n]).nonzero()
cut = int(different[0, 0]) if len(different) else n
print(json.dumps({"name": "coding-base-v1", "format": "openai", "request": request,
                  "prefix_tokens": cut, "ttl_seconds": 300}))
PY

curl -sS http://127.0.0.1:8000/v1/cache/prefixes \
  -H 'Content-Type: application/json' --data-binary @agent-prefix.json
curl -sS -X POST http://127.0.0.1:8000/v1/cache/prefixes/coding-base-v1/warm
curl -sS http://127.0.0.1:8000/v1/cache/prefixes/coding-base-v1
```

Use the model identifier returned by `/v1/models` in actual requests. If you access FreeToken
through an authenticated local proxy, include that proxy's authorization header. Prefix controls
use the same access behavior as the existing local generation routes. Replace the example instructions with your real system
prompt; keep its text, tools and template settings identical across registration and generation.
Registration returns the server's exact token hash and aligned length, so clients can verify
their boundary against the active tokenizer. The live benchmark performs this hash check.

Explicit warming is optional. If four matching agents arrive cold, the server queues one
internal preparation and holds the four requests until its final checkpoint is settled. When
already resident, ordinary cache admission reuses the prefix immediately. Preparation yields
between chunks so waiting unrelated requests and active decoding can run. Its unfinished tokens
remain reserved while other requests are admitted. Under insufficient capacity, a blocked new
request cannot prevent an already admitted preparation from finishing.

## Control API

Responses have `status`, `result` and `error` fields. Registering stores bounded source tokens;
it does not run inference. Warm returns HTTP 202 after queueing or recognizing an existing hit;
poll get/list for readiness. Timeouts do not roll back an accepted command: inspect the name
before retrying. Re-registering the same name and tokens is idempotent.

| Method and path | Action |
| --- | --- |
| `POST /v1/cache/prefixes` | Register `{name, format, request, prefix_tokens?, ttl_seconds?}`. `format` is `openai` or `anthropic`. |
| `GET /v1/cache/prefixes` | List aliases, actual unique resident pages/snapshots and registry totals. |
| `GET /v1/cache/prefixes/{name}` | Inspect aligned length, GPU/parked tokens, preparation work and last failure. |
| `POST /v1/cache/prefixes/{name}/warm` | Queue one preparation or restore for the exact checkpoint. |
| `DELETE /v1/cache/prefixes/{name}` | Remove the alias and its optional preference. Active requests retain their references. |
| `PUT /v1/cache/prefixes/settings` | Set `{ "max_retained_bytes": 1073741824 }`, measured across unique held pages and snapshots. |

States are `registered`, `warming`, `ready`, `evicted` or `error`. `ready` means the complete
checkpoint is currently on the GPU. `parked_tokens` reports an exact checkpoint in the enabled
local store, separately from GPU readiness. A stored entry may still fail validation or be
evicted before use; restore falls back to cold preparation. SSD restore keeps the existing
profitability threshold, so very short disk checkpoints may be recomputed.

`forwarded_tokens` counts settled internal preparation tokens; `restored_tokens` counts restored
prefix tokens. `preparations` counts jobs that began admission to a forward; `followers` counts
requests held for preparation. The list totals deduplicate physical pages shared by parent and
child presets. `private_state_bytes_per_request` reports the three mutable state slots reserved
for each admitted hybrid request. Normal usage reporting remains opt-in with
`--enable-cache-report`, and internal preparation never counts as generated output.

## Retention and invalidation

- Up to 64 aliases and 16 MiB of unique int32 source tokens are retained by the registry.
  Duplicate aliases share one entry. Deleted entries with an unfinished preparation count
  against the same entry/source limits until the preparation drains.
- Names contain letters, digits, dots, colons, underscores or hyphens, up to 128 characters,
  starting with a letter or digit. A name cannot silently change to different tokens; use a
  new versioned name or delete the old one first.
- The requested prefix must include one whole page, leave context for the agent's continuation,
  and fit the current KV pool with at least one page left. The unaligned tail is processed per
  request. A 25,000-token prompt with 64-token pages shares 24,960 tokens and leaves 40. An
  exactly 24,960-token prompt shares 24,896 tokens and processes its final 64 tokens per agent.
- Preference TTL defaults to 300 seconds and accepts 0–86,400 seconds. It applies to the shared
  entry's GPU preference, not to alias lifetime; the most recent registration sets that entry's
  TTL. A warm/use refreshes the preference. Zero disables lasting retention.
- The initial preference budget is the smaller of 1 GiB and half the allocated KV/state pool
  bytes. Brief handoff references can exceed it while prepared followers acquire request
  references. Registry references yield under KV/state pressure at admission and allocation;
  active request references remain protected. A full request table or exhausted token budget
  does not erase retention. Under insufficient capacity, sharing is best effort and work can be repeated.
- A pool rebuild, including `--kv-dynamic` growth or shrink, releases registry handles before
  replacing pools. Source tokens survive and readiness is looked up again. Registration uses
  the current pool and context limits, refreshed after successful resizing. A failed rebuild
  terminates waiting agents and refuses further prefix commands until the server restarts.
  Model/server restarts discard aliases; re-register them. This first version does not persist the registry. Existing SSD checkpoints retain their usual
  model/layout fingerprint and checksum rules.
- Deleting an alias does not wipe a checkpoint other requests can reuse. Deleting during
  preparation allows that bounded job to drain. Cancelling one waiting agent removes only its
  own request. If preparation is rejected, waiting requests return to ordinary admission once.
- Picture/private requests cannot be registered or routed through this shared preparation path.
- Expected tokenization or chat-template validation failures return HTTP 400 (`invalid`),
  including rejected role ordering. Unexpected worker errors remain HTTP 500 (`failed`).

Register a skill as a complete ordered prefix such as `base + coding role + review skill`.
That preset can reuse its base checkpoint, but a separately cached skill cannot be inserted
after arbitrary preceding text: recurrent and attention state depend on what came before it.

The existing parking settings remain `--kv-park off|ram|ssd`. RAM and SSD are separate modes;
this implementation does not introduce a combined RAM/SSD tier. GPU sharing works with parking
off. See the README's parking section for local budgets, persistence and restore behavior.

## Validation

CPU tests exercise four-way physical page sharing, private state ownership, intermediate chunk
lifetimes, final drain without output, restore/rebuild, cancellation and API/queue transport.
They do not establish GPU kernel correctness or real-model speed.

On an idle local validation server with `--enable-cache-report`, run:

```bash
PYTHONPATH=python python scripts/bench/agent_prefix_live.py \
  --model-path /path/to/served/model --tokens 25000 --agents 1 2 4 8 \
  --output agent-prefix-results.json
```

The benchmark records streaming time to first token, input/cache usage, preparation tokens,
actual resident page totals, answer markers and server identity. Cold baseline and coalesced
trials use fresh early nonces; warm trials reuse the registered prefix with different private
tasks. It never clears caches, changes server settings or restarts the process, and deletes only
its own aliases. `FREETOKEN_API_KEY` supplies a bearer header for an authenticated proxy. Missing cache hits fail
the acceptance check rather than being reported as successful sharing. Failed answer markers or
missing first-token timing leave the report incomplete; lower latency alone does not establish
correct model output.
