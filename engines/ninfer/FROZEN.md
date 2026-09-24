# NInfer (frozen copy)

- Source: https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile (a fork of Neroued/ninfer that
  adds the QUASAR binding; includes upstream through ce7dee50)
- Commit: d4bc75dbc7066109c3d9692ed564e5904a849ba0
- Copied: 2026-09-24, without .git and build/
- License: Apache-2.0 (LICENSE kept)
- Runs: quasar-27b (DFlash2 K7) — this is the reason this runtime is frozen here. It should
  also run fable-27b and twin-27b; the live check comparing it against the upstream build
  (Steps 1-2 of task-6-brief.md) is pending a live session. If mobile falls short of 95% of
  upstream's tok/s on either, or fails to load/answer, a second copy is added as
  `engines/ninfer-upstream/` (Neroued/ninfer at f76e19c0) and `engines/README.md` is updated
  accordingly.
- No git submodules: `git submodule status` on the clone printed nothing.
- No FetchContent/ExternalProject: `CMakeLists.txt` and the whole tree have no such calls;
  `third_party/{spdlog,utf8proc,nlohmann,cpp-httplib}` are vendored directly as plain source
  trees the root CMakeLists.txt `add_subdirectory()`s or include-paths into the build. Nothing
  is fetched at CMake configure/build time.

## Left out

Three test/tool fixture files under `tools/freq_corpus/fixtures/ranking/` (each ~12 MB),
excluded from the rsync copy because they are frequency-ranking fixtures for the Python
model-conversion tooling under `tools/convert/` (used by `draft_head.py` recipes and by the
Python test `tests/convert/qwen3_6_35b_a3b/test_draft_head.py`), not inputs to the CMake build
that produces `apps/ninfer-serve`. Confirmed by grepping every `CMakeLists.txt` in the tree:
`tools/` is never `add_subdirectory()`-ed or referenced, so `build.sh` never touches them.

- `tools/freq_corpus/fixtures/ranking/accept.heldout.counts.i64` (12 MB)
- `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` (12 MB)
- `tools/freq_corpus/fixtures/ranking/ranking.heldout.counts.i64` (12 MB)

If the Python conversion tooling or its tests are ever needed from this frozen copy, re-fetch
them from the source repo at the pinned commit:

```bash
tmp=$(mktemp -d)
git clone -q https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile.git "$tmp"
git -C "$tmp" checkout -q d4bc75dbc7066109c3d9692ed564e5904a849ba0
cp "$tmp"/tools/freq_corpus/fixtures/ranking/{accept.heldout.counts.i64,ranking.train.counts.i64,ranking.heldout.counts.i64} \
   engines/ninfer/tools/freq_corpus/fixtures/ranking/
```

Their sibling manifest/report JSON files (small) were kept, along with every other file in the
tree; a `diff -rq` of this copy against the temp clone (excluding `.git` and `build/`) shows no
other differences.

## Our patches

| Patch | What | Files |
|---|---|---|
