# NInfer (frozen copy)

- Source: https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile (a fork of Neroued/ninfer that
  adds the QUASAR binding; includes upstream through ce7dee50)
- Commit: d4bc75dbc7066109c3d9692ed564e5904a849ba0
- Copied: 2026-09-24, without .git and build/
- License: Apache-2.0 (LICENSE kept)
- Runs: quasar-27b (DFlash2 K7) only. Live check on the serving box: this build refused the
  Fable and Twin artifacts at startup with `FATAL server failed during startup | artifact magic
  is not NInfer v2` — those artifacts are NInfer v3, converted with upstream Neroued/ninfer at
  f76e19c0fbd026c86f46005acf2c80c54084bade, and this mobile fork's artifact loader only reads
  v2. Per the spec's fallback, a second copy is frozen at `engines/ninfer-upstream/` (that same
  upstream commit) to run fable-27b and twin-27b; see `engines/README.md` and
  `engines/ninfer-upstream/FROZEN.md`.
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
