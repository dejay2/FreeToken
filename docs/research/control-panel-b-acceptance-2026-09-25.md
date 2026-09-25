# Control panel stage B: live acceptance (2026-09-25)

Box: Windows 11 + WSL `vllm`, RTX 5090, helper 2.1.0 on `feat/control-panel-b` (c0b9872 plus the
backup-prune fix), llama-swap frozen build from main (7a7f0ea). Nothing was loaded during the test.
Driven through the real page over the tailnet (chrome-devtools-axi), screenshots in `img/stage-b-*`.

| Step | What happened | Screenshot |
|---|---|---|
| Open the wizard | three steps, "On this PC" / "Download from Hugging Face" | `stage-b-01-add-wizard.png` |
| A symlink to an existing artifact | detected as NInfer v3 (upstream runtime), 20.0 GB, "already in the list as Twin 27B" — refused, Save off | `stage-b-02-add-file-detected.png` |
| Unsupported folders (`Qwen3DSparkModel`) | "not supported by your engines", identity step hidden, Save off | `stage-b-03-unsupported.png` |
| Add a folder on the PC | `~/models/Qwen3.8-27B-NVFP4-RTX5090` → `Qwen3_5ForConditionalGeneration` for FreeToken, 16.7 GB, id suggested, RAM 17 GB; added; Pi `freetoken-local` + `enabledModels` gained it, other 4 providers untouched; timestamped backups of both Pi files written; switcher `/v1/models` listed it | `stage-b-04`, `stage-b-05` |
| Link plan | `https://huggingface.co/Qwen/Qwen3-0.6B` → model folder 1.4 GB into `~/models/Qwen3-0.6B`, 2 of 7 files checked against published checksums | `stage-b-06-link-plan.png` |
| Cancel mid-download | "Download cancelled. Its partial files were deleted."; no `.incoming-*` and no target folder left on disk | `stage-b-08-cancelled.png` |
| Full download | 100 %, 1.4 GB, "2 file(s) matched the published checksums"; detected as `Qwen3ForCausalLM` for FreeToken, id `qwen3-0.6b`, RAM 2 GB | `stage-b-10-downloaded-detected.png` |
| Add it | listed; Pi and switcher updated | `stage-b-11-added-hf.png` |
| Remove with "delete files" ticked | box starts unticked; note names the folder and its settings profile; button reads "Remove and delete files"; "Removed Qwen3-0.6B (FreeToken). Its files were deleted. Pi updated."; folder gone | `stage-b-13`, `stage-b-14` |
| Remove without ticking | "Removed Qwen3.8-27B-NVFP4-RTX5090 (FreeToken). Pi updated."; folder kept (17 GB) | `stage-b-15-removed-folder-kept.png` |
| End state | Pi `models.json` and `settings.json` parse identical to the copies taken before the test; switcher back to its 5 models; Twin's real artifact untouched | — |

Found live and fixed before merge: the Pi backup prune counted hand-made backups
(`settings.json.bak-before-*`, several exist on the box) toward the 20 kept and could delete them;
it now prunes only its own timestamped copies (`test_pruning_never_touches_hand_made_backups`).
