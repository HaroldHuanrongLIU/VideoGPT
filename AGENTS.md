# Repository Guidelines

## Project Structure & Module Organization

`videogpt/` is the installable PyTorch package. Core model code lives in `vqvae.py`, `gpt.py`, `attention.py`, `resnet.py`, `data.py`, and `utils.py`; FVD support is under `videogpt/fvd/`. `scripts/` contains command-line entry points for training, sampling, evaluation, and dataset conversion. Dataset converters are grouped under `scripts/preprocess/bair/` and `scripts/preprocess/ucf101/`. SurgWMBench-specific tests live in `tests/`. `notebooks/` contains the Colab/demo workflow, and `VideoGPT.png` is the README architecture asset.

## Build, Test, and Development Commands

Install dependencies and the local package from the repository root with uv:

```bash
uv sync
uv run python -m pytest -q tests
```

Use the help output before changing CLI arguments:

```bash
uv run python scripts/train_vqvae.py -h
uv run python scripts/train_videogpt.py -h
uv run python scripts/sample_videogpt.py -h
```

Run a lightweight syntax/import check before submitting changes:

```bash
uv run python -m compileall videogpt scripts tests
```

Dataset examples from the README are shell scripts, for example `sh scripts/preprocess/bair/create_bair_dataset.sh datasets/bair`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and follow the existing import style: standard library, third-party packages, then local modules. Classes use `CamelCase` (`VideoGPT`, `VQVAE`); functions, variables, and CLI flags use `snake_case`. Keep tensor layout comments explicit when crossing data boundaries, especially `THWC`, `CTHW`, and `BCTHW`. Prefer small, local changes over broad refactors.

## Testing Guidelines

Use `uv run python -m pytest -q tests` for focused coverage. For new functionality, add tests under `tests/` with names like `test_data.py` or `test_vqvae.py`. Use tiny synthetic tensors, temporary HDF5 files, or minimal video fixtures instead of requiring full BAIR/UCF downloads. At minimum, run `uv run python -m compileall videogpt scripts tests` and relevant script `-h` commands.

## Commit & Pull Request Guidelines

Recent history uses short verb-led messages such as `Update README.md`, `Fix Colab downloading`, and `Delete demo.py`. Keep commits similarly concise and scoped. Pull requests should describe the behavior changed, list validation commands run, mention dataset/checkpoint assumptions, and include sample outputs or screenshots when sampling, notebook, or media behavior changes.

## Data & Artifact Hygiene

Do not commit datasets, checkpoints, generated videos, downloaded model weights, or generated `metadata_*.pkl` cache files. Keep machine-specific paths in examples or command arguments, not in package defaults.
