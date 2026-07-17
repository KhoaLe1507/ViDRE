# ViDRE: Text-to-Keyframe Video Retrieval Research Project

ViDRE is a research-oriented system for retrieving relevant video moments from
natural language queries. Given a text query, the system searches a large video
collection and returns ranked keyframes, timestamps, and video identifiers.

## Research Motivation

Text-to-keyframe retrieval is difficult when the query describes an action. A
query such as "a person opens a refrigerator" is temporal, but a keyframe is a
single static image. A single frame may contain only weak evidence of the action:
the person, the object, the pose, or the scene, but not the full motion.

Early experiments showed that direct global keyframe search was not enough. The
main failure mode was that the system often retrieved the wrong video before it
even had a chance to rank the correct keyframe. Once the correct video was
forced as the search scope, temporal keyframe recall became very high. This led
to the central research direction of the project:

> retrieve the correct candidate video first, then perform fine-grained keyframe
> ranking inside that smaller video set.

## Research Questions

- How well do image-text models such as OpenCLIP and BEiT-3 work for
  action-centric text-to-keyframe retrieval?
- Is the main bottleneck keyframe ranking, or retrieving the correct video
  candidate first?
- Can video-first retrieval and query-conditioned frame pooling improve action
  query retrieval without using expensive full video-language models?
- How much does query specificity affect retrieval quality on a dense-captioning
  dataset adapted for global retrieval?

## Contributions

- Built an end-to-end offline and online retrieval pipeline for text-to-keyframe
  search over Charades-TimeLens videos.
- Integrated TransNetV2 for shot/keyframe processing, BEiT-3 and OpenCLIP for
  multimodal embeddings, Co-DETR for object detection, Gemini for query
  expansion and query rewriting experiments, and Stable Diffusion for visual
  query generation experiments.
- Deployed the pipeline on Modal GPU containers with a production-like storage
  layer: vector search for embeddings, relational metadata storage, and object
  storage for video/keyframe assets.
- Diagnosed the core failure mode of action-query retrieval: global video
  retrieval is the bottleneck, while keyframe selection becomes much easier once
  the correct video is in scope.
- Implemented video-first retrieval variants, including video/window
  aggregation and a query-conditioned frame pooling prototype.

## Dataset

The benchmark is based on Charades-TimeLens, derived from Charades-style indoor
activity videos. The original query set is action-centric and often concise,
which makes it useful for analyzing the mismatch between action queries and
static keyframes.

Current processed scale:

- Videos: about 1,014 verified videos
- Keyframes: about 6.3K indexed keyframes
- Original action queries: 2,594 queries
- Scene/moment query subset: 1,000 rewritten queries used for diagnostic
  retrieval analysis

## Method

### Offline Phase

The offline phase prepares the searchable index:

1. Read videos from the dataset directory.
2. Detect shots and extract representative keyframes with TransNetV2.
3. Encode keyframes with BEiT-3 and OpenCLIP.
4. Run object detection with Co-DETR for optional object-aware filtering.
5. Store vectors in Zilliz/Milvus.
6. Store structured metadata in CockroachDB.
7. Store video/keyframe assets in object storage.

### Online Phase

The online phase receives a text query and returns ranked keyframes. The project
supports several retrieval modes:

- `openclip_user_query_baseline`: direct OpenCLIP text-to-keyframe retrieval.
- `beit3_user_query_baseline`: direct BEiT-3 text-to-keyframe retrieval.
- `textual_query_baseline`: OpenCLIP + BEiT-3 retrieval with fusion.
- `full`: original multi-branch pipeline with query expansion, visual query
  generation, vector search, fusion, and optional object filtering.
- `beit3_video_mean_top3_baseline`: video-first retrieval by aggregating
  keyframe scores per video.
- `beit3_window_mean_top3_baseline`: window-level aggregation over keyframes.
- Query-conditioned video-first reranking: reranks candidate videos by comparing
  the query against multiple frame embeddings from each video.
- All-video query-conditioned reranking: samples multiple keyframes per video
  and reranks videos before selecting final keyframes.
- `openclip_llm_scene_rewrite_baseline`: query-only rewrite into a more
  visually grounded form, followed by OpenCLIP retrieval.

### Query-Conditioned Frame Pooling

The query-conditioned frame pooling prototype compares one query embedding against multiple
keyframe embeddings from a candidate video. Instead of representing a video by a
fixed mean-pooled vector, it computes query-dependent attention weights over the
video's frame embeddings. Frames that are more relevant to the query receive
larger weights, producing a video representation that depends on the query.

This is designed to help action queries because different actions may require
attention to different frames in the same video.

Important note: the current implementation is an online-only prototype unless a
trained query-conditioned pooling checkpoint is supplied. It should not be read
as a fully trained video retrieval model.

## Evaluation

The main metrics are:

- Keyframe Recall@1, Recall@5, Recall@10
- Video Recall@1, Recall@5, Recall@10
- Mean online latency per query

A predicted keyframe is counted as correct when it belongs to the ground-truth
video and its timestamp falls inside the ground-truth temporal span. Video
recall ignores timestamp and checks only whether the ground-truth video appears
in the top-k results.

## Key Findings

### 1. Direct action-query keyframe retrieval is weak

On the original action-query benchmark, direct keyframe retrieval performs
poorly because many action queries are too short and too temporal for a single
static keyframe.

| Setting | Query set | Keyframe R@1 | Keyframe R@5 | Keyframe R@10 |
| --- | --- | ---: | ---: | ---: |
| Direct global keyframe baseline | Original action queries | 0.017 | 0.043 | 0.064 |

### 2. The main bottleneck is retrieving the correct video

Diagnostics showed that BEiT-3 and OpenCLIP embeddings were not broken:
self-retrieval of indexed keyframes achieved top-1 accuracy of 1.0. When search
was constrained to the correct video, in-video temporal Recall@10 reached about
1.0. This indicates that the hardest part is not always choosing the best
keyframe inside a correct video, but retrieving the correct video globally.

### 3. Video-first retrieval helps, but action queries remain difficult

On the original action-query setting, video-first aggregation and
query-conditioned reranking improved the video retrieval signal, but the task
remained challenging due to underspecified action labels.

| Setting | Queries | Keyframe R@1 | Keyframe R@5 | Keyframe R@10 | Video R@1 | Video R@5 | Video R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BEiT-3 query-conditioned video-first prototype | 1,000 | 0.027 | 0.086 | 0.125 | 0.045 | 0.132 | 0.207 |

### 4. Visually grounded scene/moment queries are much easier to retrieve

When the query includes more visual scene, object, and moment cues, retrieval
quality increases sharply. This suggests that the original dense-captioning
queries can be ambiguous for global retrieval, where many videos may match the
same short action phrase but only one is labeled as ground truth.

| Setting | Queries | Keyframe R@1 | Keyframe R@5 | Keyframe R@10 | Video R@1 | Video R@5 | Video R@10 | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCLIP direct baseline on scene/moment queries | 1,000 | 0.404 | 0.694 | 0.783 | 0.584 | 0.821 | 0.897 | 42 ms |
| BEiT-3 query-conditioned video-first on scene/moment queries | 1,000 | 0.327 | 0.512 | 0.568 | 0.485 | 0.758 | 0.852 | 831 ms |

## Repository Structure

```text
configs/
  default.yaml                Main experiment and model configuration
docs/
  Project notes and experiment documentation
external/
  Co-DETR/
  TransNetV2/
  unilm/
scripts/
  apply_cockroach_migration.py
  create_zilliz_collection.py
  validate_model_assets.py
src/
  eval/                       Evaluation, metrics, query loading
  models/                     Model loaders and clients
  offline/                    Offline indexing phase
  online/                     Search pipeline, fusion, query-conditioned pooling prototypes
  storage/                    Zilliz, CockroachDB, object storage clients
modal_app.py                  Modal GPU entrypoints and diagnostics
```

## Setup

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Validate local model assets:

```powershell
python scripts/validate_model_assets.py --config configs/default.yaml
```

Required model assets are expected under:

```text
data/models/
```

The current system expects local weights for BEiT-3, OpenCLIP ViT-H/14 DFN5B,
TransNetV2, Co-DETR, and SDXL when running the full pipeline.

## Model and Dataset Downloads

Large model files and benchmark videos are not stored in this repository. Place
them under the paths expected by `configs/default.yaml`.

### Model Assets

| Component | Source | Expected local path |
| --- | --- | --- |
| BEiT-3 large ITC | [Microsoft UniLM BEiT-3 release](https://github.com/microsoft/unilm/tree/master/beit3) | `data/models/beit3-large-itc/beit3_large_itc_patch16_224.pth` and `data/models/beit3-large-itc/beit3.spm` |
| OpenCLIP DFN5B ViT-H/14 | [apple/DFN5B-CLIP-ViT-H-14](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14) | `data/models/openclip-dfn5b-vit-h-14/open_clip_pytorch_model.bin` |
| TransNetV2 | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) | `data/models/transnetv2/transnetv2-weights/` |
| Co-DETR ViT-L COCO | [zongzhuofan/co-detr-vit-large-coco](https://huggingface.co/zongzhuofan/co-detr-vit-large-coco) | `data/models/codetr-vit-large-coco/pytorch_model.pth` |
| SDXL base 1.0 | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `data/models/sdxl-base-1.0-diffusers-fp16/` |

Example download commands:

```powershell
python -m pip install -U huggingface_hub

New-Item -ItemType Directory -Force data/models/beit3-large-itc
Invoke-WebRequest https://github.com/addf400/files/releases/download/beit3/beit3_large_itc_patch16_224.pth -OutFile data/models/beit3-large-itc/beit3_large_itc_patch16_224.pth
Invoke-WebRequest https://github.com/addf400/files/releases/download/beit3/beit3.spm -OutFile data/models/beit3-large-itc/beit3.spm

hf download apple/DFN5B-CLIP-ViT-H-14 open_clip_pytorch_model.bin open_clip_config.json --local-dir data/models/openclip-dfn5b-vit-h-14

git lfs install
git clone https://github.com/soCzech/TransNetV2 external/TransNetV2
New-Item -ItemType Directory -Force data/models/transnetv2
Copy-Item -Recurse external/TransNetV2/inference/transnetv2-weights data/models/transnetv2/transnetv2-weights

hf download zongzhuofan/co-detr-vit-large-coco pytorch_model.pth --local-dir data/models/codetr-vit-large-coco

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/stable-diffusion-xl-base-1.0', local_dir=r'data/models/sdxl-base-1.0-diffusers-fp16')"
```

The full pipeline also needs the external source trees used by the local model
loaders:

```powershell
git clone https://github.com/microsoft/unilm external/unilm
git clone https://github.com/Sense-X/Co-DETR external/Co-DETR
git clone https://github.com/soCzech/TransNetV2 external/TransNetV2
```

After downloading, verify the expected files:

```powershell
python scripts/validate_model_assets.py --config configs/default.yaml
```

### Benchmark Dataset

The benchmark annotations and videos are downloaded from
[TencentARC/TimeLens-Bench](https://huggingface.co/datasets/TencentARC/TimeLens-Bench).
This project currently uses the Charades-TimeLens split.

Download the benchmark:

```powershell
python -m pip install -U huggingface_hub
hf download TencentARC/TimeLens-Bench --repo-type dataset --local-dir data/TimeLens-Bench
```

Extract the video shards:

```powershell
New-Item -ItemType Directory -Force data/TimeLens-Bench/videos
Get-ChildItem data/TimeLens-Bench/video_shards -Recurse -Filter *.tar.gz | ForEach-Object { tar -xzf $_.FullName -C data/TimeLens-Bench/videos }
```

Expected Charades files:

```text
data/TimeLens-Bench/charades-timelens.json
data/TimeLens-Bench/videos/charades/*.mp4
```

Build the query-level JSON used by this project:

```powershell
python -m src.eval.build_query_samples --input data/TimeLens-Bench/charades-timelens.json --output data/TimeLens-Bench/charades-timelens-query-samples.json
```

The scene/moment query file is derived from the original benchmark queries and
sampled ground-truth frames. It is not part of the raw download:

```text
data/TimeLens-Bench/charades-timelens-query-samples-scene-moment-first1000.json
```

## Environment Variables

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
$env:GEMINI_MODEL="gemini-3.5-flash"

$env:MODEL_DIR="data/models"
$env:CONFIG_PATH="configs/default.yaml"
```

## Storage Setup

Apply the relational schema:

```powershell
python scripts/apply_cockroach_migration.py --config configs/default.yaml
```

Create the vector collection:

```powershell
python scripts/create_zilliz_collection.py --config configs/default.yaml
```

Use `--drop-existing` only when intentionally recreating the collection.

## Running the Offline Phase

Local smoke run:

```powershell
python -m src.offline.run_offline_phase --limit 1
```

Modal smoke run:

```powershell
modal run modal_app.py::run_offline_phase --limit 1
```

Detached full Modal run:

```powershell
modal run --detach --name vidre-offline-full modal_app.py::run_offline_phase
```

## Running Evaluation

Direct OpenCLIP baseline:

```powershell
modal run modal_app.py::run_openclip_baseline_evaluation --dataset-path /data/TimeLens-Bench/charades-timelens-query-samples.json --limit 1000 --require-verified-gt-video --require-gt-span-keyframe
```

Query-only scene rewrite baseline:

```powershell
modal run modal_app.py::run_openclip_scene_rewrite_baseline_evaluation --dataset-path /data/TimeLens-Bench/charades-timelens-query-samples.json --limit 1000 --require-verified-gt-video --require-gt-span-keyframe
```

Single-query debugging:

```powershell
modal run modal_app.py::search_single_query --query "a person opens a door" --online-mode openclip_user_query_baseline
```

## Diagnostics

Check whether embeddings and vector fields are internally consistent:

```powershell
modal run modal_app.py::diagnose_embedding_self_retrieval_cli --model-space openclip --limit 50 --top-k 5
modal run modal_app.py::diagnose_embedding_self_retrieval_cli --model-space beit3 --limit 50 --top-k 5
```

Analyze whether failures come from global video retrieval or keyframe selection:

```powershell
modal run modal_app.py::diagnose_retrieval_failure_modes_cli --model-space openclip --limit 100 --global-depth 1000
modal run modal_app.py::diagnose_retrieval_failure_modes_cli --model-space beit3 --limit 100 --global-depth 1000
```

## Current Limitations

- The query-conditioned pooling modules are prototypes unless trained checkpoints are
  supplied.
- The current system does not implement a full video-language model encoder.
- Action queries from dense-captioning datasets can be ambiguous in global
  multi-video retrieval because multiple videos may satisfy the same short
  action phrase.
- The project focuses on retrieval and evaluation. It does not include a
  production UI, VQA, spatial relation reasoning, or user-uploaded visual query
  support.

## Research Summary

The main lesson from ViDRE is that action-centric text-to-keyframe retrieval
should not be treated only as static image retrieval. For short action queries,
the system must first recover the correct candidate video or temporal region.
Query-conditioned frame pooling and video-first retrieval are practical ways to
use multiple weak frame-level signals before selecting the final keyframe.
