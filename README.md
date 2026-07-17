# ViDRE Text-to-Keyframe Retrieval

This repo implements the Text-to-Keyframe Retrieval scope from
`docs/text_to_keyframe_retrieval_spec.md`. The VBS paper is treated as
background only; the spec wins on conflicts.

## What is included

- Offline phase: TransNetV2 shot detection, BEiT-3 keyframe selection,
  BEiT-3/OpenCLIP embedding, Co-DETR object counts, Zilliz insert,
  CockroachDB metadata, R2 shot proxy upload, resume by verified video/shot.
- Online phase: original text branch, Gemini paraphrase branch, Stable
  Diffusion visual query branch, intra-branch RRF, inter-branch Weighted RRF,
  post-fusion hard Object Filter.
- Evaluation: Recall@1, Recall@5, Recall@10, Mean_Time_Latency, plus
  `metrics.json`, `per_query_results.jsonl`, and `run_manifest.json`.

No UI, temporal search, VQA, spatial relation, or user-uploaded visual query is
implemented.

## Environment variables

```powershell
$env:ZILLIZ_URI="..."
$env:ZILLIZ_TOKEN="..."
$env:ZILLIZ_COLLECTION="vidre_keyframes"

$env:COCKROACH_DATABASE_URL="postgresql://..."

$env:R2_ACCOUNT_ID="..."
$env:R2_ACCESS_KEY_ID="..."
$env:R2_SECRET_ACCESS_KEY="..."
$env:R2_BUCKET="..."
$env:R2_ENDPOINT_URL="https://<account>.r2.cloudflarestorage.com"

$env:GEMINI_API_KEY="..."
$env:GEMINI_MODEL="..."

$env:MODEL_DIR="data/models"
$env:CONFIG_PATH="configs/default.yaml"
```

## Model assets

Run:

```powershell
python scripts/validate_model_assets.py --config configs/default.yaml
```

Current local assets include BEiT-3, Co-DETR checkpoint, SDXL fp16, and
TransNetV2 weights. The OpenCLIP directory currently contains config/tokenizer
files but no detected weight file. Add one supported weight file under
`data/models/openclip-dfn5b-vit-h-14`, for example:

- `pytorch_model.bin`
- `model.safetensors`
- `open_clip_pytorch_model.bin`
- `open_clip_pytorch_model.safetensors`

The expected model is OpenCLIP ViT-H/14 DFN5B with 1024-dim embeddings.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Co-DETR needs a CUDA-matched MMDetection 2.x stack. If `mmdet` import fails,
install the matching `mmcv-full` build for your CUDA/PyTorch version and review
`external/Co-DETR/requirements.txt`.

## Storage setup

Apply CockroachDB schema:

```powershell
python scripts/apply_cockroach_migration.py --config configs/default.yaml
```

Create the Zilliz collection with two vector fields:

```powershell
python scripts/create_zilliz_collection.py --config configs/default.yaml
```

Use `--drop-existing` only when you intentionally want to recreate the
collection.

## Offline phase

```powershell
python -m src.offline.run_offline_phase
```

For a smoke run:

```powershell
python -m src.offline.run_offline_phase --limit 1
```

The code has a `limit` argument in the Python API and Modal function. The CLI
module currently uses default config; call the function from Python for custom
limits or use Modal.

Raw video deletion is implemented but disabled by default in
`configs/default.yaml` via `offline.delete_raw_after_verified: false` to avoid
destroying the local Charades dataset. Set it to `true` only for the production
R2 raw-video lifecycle after verifying vectors, metadata, object counts, shot
proxy upload, and mapping integrity.

## Online evaluation

```powershell
python -m src.eval.run_evaluation --config configs/default.yaml --latency-mode cache_miss_full_online
```

Cache-hit retrieval-only mode:

```powershell
python -m src.eval.run_evaluation --config configs/default.yaml --latency-mode cache_hit_retrieval_only
```

Do not mix latency modes in one reported number. The output directory is:

```text
outputs/eval/<eval_run_id>/
  metrics.json
  per_query_results.jsonl
  run_manifest.json
```

`Mean_Time_Latency` is measured from text query receipt through final ranked
list after Weighted RRF and Object Filter. It excludes offline work, thumbnail
cutting, UI rendering, and returning shot video to a client.

## Modal entrypoints

`modal_app.py` exposes:

- `run_offline_phase`
- `run_online_evaluation`
- `search_single_query`

The Modal GPU settings are initial proposed settings from the spec. They are not
benchmarked latency or VRAM claims. Benchmark before reporting production
numbers or optimizing GPU size.

