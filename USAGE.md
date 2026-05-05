# Usage

This document covers the SurgWMBench 20-anchor VideoGPT workflow: create the
standard uv environment, train the 20-frame VQ-VAE, then train a 5-frame
conditioned VideoGPT model that predicts anchor frames 6-20.

## Environment

Use the locked uv environment from the repository root:

```bash
uv sync
```

Verify the synced environment:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
uv run python -m pytest -q tests
```

The default dataset root used by the scripts is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

The training path uses official manifests only. Each sample loads the 20
human-anchor frames identified by `sampled_indices`.

## Changing Dataset Path

Set a different SurgWMBench location with `--dataset-root` in every train or
eval command:

```bash
--dataset-root /path/to/SurgWMBench
```

Manifest paths stay relative to that dataset root. For example, if your dataset
is under `/data/SurgWMBench`, use:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /data/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl
```

Do not edit the official manifest files or create random splits. Keep using
`manifests/train.jsonl`, `manifests/val.jsonl`, and `manifests/test.jsonl`
from the dataset root.

## Single-GPU Training

Train the SurgWMBench 20-anchor VQ-VAE first:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --sequence_length 20 \
  --resolution 128 \
  --batch_size 4 \
  --num_workers 8 \
  --max_steps 200000 \
  --accelerator gpu \
  --devices 1
```

Then train VideoGPT with the VQ-VAE checkpoint:

```bash
uv run python scripts/train_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --vqvae <path-to-vqvae.ckpt> \
  --sequence_length 20 \
  --n_cond_frames 5 \
  --resolution 128 \
  --batch_size 2 \
  --num_workers 8 \
  --max_steps 200000 \
  --accelerator gpu \
  --devices 1
```

## Multi-GPU Training

Use Lightning DDP by setting `--devices` to the number of visible GPUs.

VQ-VAE multi-GPU:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --sequence_length 20 \
  --resolution 128 \
  --batch_size 4 \
  --num_workers 8 \
  --max_steps 200000 \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp
```

VideoGPT multi-GPU:

```bash
uv run python scripts/train_surgwmbench_videogpt.py \
  --dataset-root /mnt/data/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --vqvae lightning_logs/version_1/checkpoints/epoch=52-step=5000.ckpt \
  --sequence_length 20 \
  --n_cond_frames 5 \
  --resolution 128 \
  --batch_size 2 \
  --num_workers 8 \
  --max_steps 200000 \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp_find_unused_parameters_false
```

`--batch_size` is per process/GPU under DDP. Adjust it for GPU memory.

## Evaluation

Evaluate one VideoGPT checkpoint on the test manifest. The script generates the
future anchor sequence and reports metrics for horizons 5, 10, and 15:

```bash
uv run python scripts/eval_surgwmbench_videogpt.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/test.jsonl \
  --ckpt <path-to-videogpt.ckpt> \
  --horizons 5 10 15 \
  --output-dir outputs/surgwmbench_videogpt_eval
```

Predictions are restored to original frame resolution before PSNR, SSIM, and
LPIPS are computed. Use `--lpips-downsample 64` for faster CPU smoke checks.

## Smoke Commands

CPU smoke training with one clip:

```bash
uv run python scripts/train_surgwmbench_vqvae.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --sequence_length 20 \
  --resolution 32 \
  --batch_size 1 \
  --num_workers 0 \
  --max-clips 1 \
  --max_steps 1 \
  --accelerator cpu \
  --devices 1 \
  --limit_train_batches 1 \
  --limit_val_batches 1 \
  --num_sanity_val_steps 0 \
  --embedding_dim 8 \
  --n_codes 32 \
  --n_hiddens 16 \
  --n_res_layers 1 \
  --downsample 4 4 4
```
