# NInfer upstream (frozen copy)

- Source: https://github.com/Neroued/ninfer.git (upstream; MirkoCovizzi/ninfer-rtx5090-mobile in
  `engines/ninfer/` is a fork of this that adds the QUASAR binding)
- Commit: f76e19c0fbd026c86f46005acf2c80c54084bade
- Copied: 2026-09-24, without .git and build/
- License: Apache-2.0 (LICENSE kept)
- Runs: fable-27b and twin-27b (NInfer v3 artifacts); the QUASAR runtime in engines/ninfer reads
  only v2. Live check on the serving box: `engines/ninfer`'s build refused both Fable and Twin
  artifacts with `FATAL server failed during startup | artifact magic is not NInfer v2` (those
  artifacts were converted with this upstream commit's v3 tooling), so both copies are frozen —
  see `engines/README.md`.
- No git submodules: `git submodule status` on the clone printed nothing.
- No FetchContent/ExternalProject: `CMakeLists.txt`, `cmake/Dependencies.cmake`, and every other
  `CMakeLists.txt`/`.cmake` file in the tree have no such calls;
  `third_party/{spdlog,utf8proc,nlohmann,cpp-httplib,llama-jinja}` are vendored directly as plain
  source trees the root CMakeLists.txt `add_subdirectory()`s or include-paths into the build.
  Nothing is fetched at CMake configure/build time.

## Left out

Three test/tool fixture files under `tools/freq_corpus/fixtures/ranking/` (each ~12 MB),
excluded from the rsync copy because they are frequency-ranking fixtures for the Python
model-conversion tooling under `tools/convert/`, not inputs to the CMake build that produces
`apps/ninfer-serve`. Confirmed by grepping every `CMakeLists.txt` in the tree: `tools/` is never
`add_subdirectory()`-ed or referenced, so `build.sh` never touches them. Same rule as
`engines/ninfer/FROZEN.md`.

- `tools/freq_corpus/fixtures/ranking/accept.heldout.counts.i64` (12 MB)
- `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` (12 MB)
- `tools/freq_corpus/fixtures/ranking/ranking.heldout.counts.i64` (12 MB)

Their sibling manifest/report JSON files (small) were kept, along with every other file in the
tree; a `diff -rq` of this copy against the temp clone (excluding `.git` and `build/`) shows no
other differences.

If the Python conversion tooling or its tests are ever needed from this frozen copy, re-fetch
them from the source repo at the pinned commit:

```bash
tmp=$(mktemp -d)
git clone -q https://github.com/Neroued/ninfer.git "$tmp"
git -C "$tmp" checkout -q f76e19c0fbd026c86f46005acf2c80c54084bade
cp "$tmp"/tools/freq_corpus/fixtures/ranking/{accept.heldout.counts.i64,ranking.train.counts.i64,ranking.heldout.counts.i64} \
   engines/ninfer-upstream/tools/freq_corpus/fixtures/ranking/
```

## Our patches

| Patch | What | Files |
|---|---|---|
