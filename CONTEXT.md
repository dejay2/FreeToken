# Project glossary

## Desktop-assisted Windows engine

The native-Windows setup that runs source from this fork while reusing the Python program, Windows libraries, and CUDA files installed by FreeToken Desktop. FreeToken Desktop itself is not opened or changed.

_Avoid:_ standalone Windows package, Desktop-managed server

## Known-good text rollback

The unchanged local checkout `D:\FreeToken-ple-mmap`, branch `windows-ple-mmap`, at commit `14ee7b0`, proven with text, reasoning, tools, SSD-backed PLE, and 262,144 usable prompt tokens. It remains the text-only rollback even though the fork's public `windows-ple-mmap` branch later advanced through accepted still-picture commit `c5876d5`.

_Avoid:_ old server, public branch, production server

## SSD-backed PLE

The model's 47.7 GiB lookup table read through file-backed memory mapping from the SSD using `--ple-backend mmap`. Windows may keep recently used pieces in normal memory, but the complete table is not copied there at startup.

_Avoid:_ disk offload, RAM-loaded PLE

## Full-context allocation

A FreeToken memory reservation that leaves 262,144 prompt and reply tokens usable after its reserved 64-token page. Picture work must retain this allocation from the first live test.

_Avoid:_ 256K setting, maximum tokens

## Picture path

The first stage of visual input support for Qwen3.8 Flash Next. It accepts still pictures carried in a request, fetched from a web address, or read from a local file, then joins the model's picture understanding with its text processing.

_Avoid:_ full multimedia support, video path

## Broad picture source access

The user-approved local policy that permits any readable local picture and any web address, subject to size, time, and valid-picture checks. This is safe only while the server remains restricted to `127.0.0.1` and is not suitable for a shared or public server.

_Avoid:_ sandboxed picture access, public-safe fetching

## Picture prompt chunking

The picture path reads one complete still picture request through repeated prompt-loading steps, each no larger than the live 8,192-token work budget. The complete picture-bearing prompt may use the available combined model context; 8,192 is a per-step work size, not a picture or model context limit.

_Avoid:_ picture token limit, automatic picture shrinking, full-context single pass

## Video stage

A later stage, begun only after the picture path passes its checks. It will add native video through FreeToken's direct local address; pi currently has no video message type, so video needs its own design.

_Avoid:_ picture path, frame sampling

## Vision weights

The checkpoint's 333 picture-reading tensors, totaling about 0.84 GiB. Picture mode loads them from the existing checkpoint; they are not a separate download.

_Avoid:_ missing weights, separate vision download

## Layer-streamed picture reader

The accepted placement at `c5876d5` that keeps all 333 picture tensors (897,862,112 bytes) in ordinary CPU memory and copies one picture component or transformer block at a time into one reusable RTX 5090 workspace. It is not full-CPU picture execution: picture hidden states and each staged layer's calculations run on the GPU. The temporary allocator segment is released after each picture attempt because live measurement showed retaining about 631 MiB reduced subsequent text speed. Final acceptance restored 4,063 cached experts, preserved exactly 262,144 usable tokens, averaged 52.10 output tokens/second after picture work, and kept the retained screenshot below six seconds. Jay later authorized publishing this accepted implementation to the fork's `windows-ple-mmap` branch.

_Avoid:_ CPU vision, full-CPU vision, permanent GPU vision tower

## MTP head

The checkpoint's one-layer multi-token prediction helper, containing 31 BF16 tensors totaling 5,214,301,696 bytes (about 4.86 GiB). The current FreeToken Qwen loader deliberately skips it, so the accepted server produces one confirmed token per decoding step and spends no working memory on this helper.

_Avoid:_ PLE, n-gram table, already-loaded predictor

## Private MTP feasibility spike

The completed shadow-only investigation that proved the checkpoint's native MTP guesses and isolated target checking on this RTX 5090. It did not let guesses change answers and found that its safe prompt-style checker took about 1.29 seconds, so its timing does not represent a production-shaped MTP server.

_Avoid:_ MTP release, full MTP implementation, public MTP branch

## Private fast-checker spike

A follow-up investigation that replaces the slow prompt-style target check with a fixed one-to-four-token generation check shaped like production speculative serving. It remains private and cannot change client answers; complete MTP serving requires separate approval after the measured combined rate exceeds the accepted server's roughly 50 output tokens per second.

_Avoid:_ full MTP server, user-facing MTP, fast MTP release

## Private fast-checker repair

The staged private follow-up that first restores scratch-page isolation and trustworthy state checks, then repairs timing, random-stream, specialist-movement, and CUDA-graph evidence before a guarded correctness request and conditional full matrix. It remains checker-only and cannot choose client output.

_Avoid:_ speculative serving repair, MTP release, isolation-only patch

## Scratch target page lease

A temporary allocation of free target K/V pages used only by one private verifier transaction. Leased page bases must be page-aligned, disjoint from the live request, restored in exact allocator order, and never substituted with writes to live K/V followed by rollback.

_Avoid:_ live-tail backup, cache rollback, shared final page

## Speculative serving

The future user-facing loop that makes MTP guesses, checks them with the target, keeps an accepted prefix, corrects a rejection under normal sampling, and returns the resulting tokens to the client. It is not approved by the private fast-checker spike.

_Avoid:_ shadow checking, MTP feasibility probe
