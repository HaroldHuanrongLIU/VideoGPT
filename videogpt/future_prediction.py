from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from .surgwmbench_data import (
    DENSE_TRACK,
    SPARSE_TRACK,
    default_context_frames,
    default_prediction_horizon,
    load_annotation,
    read_jsonl,
    resolve_manifest,
    sample_windows,
    trajectory_target_for_track,
    validate_track_settings,
)
from .surgwmbench_metrics import (
    aggregate_metric_dicts,
    empty_image_metrics,
    empty_trajectory_metrics,
    image_metrics,
    trajectory_metrics,
)


PREDICTION_TASKS = ("future_frames", "future_trajectory", "future_joint")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SurgWMBench future prediction adapter for VideoGPT.")
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--prediction-task", choices=PREDICTION_TASKS, required=True)
    parser.add_argument("--data-track", choices=(SPARSE_TRACK, DENSE_TRACK), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--interpolation-method", default="linear", choices=("linear", "pchip", "akima", "cubic_spline"))
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--prediction-horizon", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--inverse-window-count-reweight", action="store_true")
    parser.add_argument("--max-sample-artifacts", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-model", choices=("none", "copy_last"), default="none")
    parser.add_argument("--frame-predictor", choices=("copy_last", "native_videogpt"), default="copy_last")
    parser.add_argument("--trajectory-predictor", choices=("constant_velocity", "copy_last"), default="constant_velocity")
    parser.add_argument("--report-horizons", nargs="*", type=int, default=None)
    parser.add_argument("--native-data-dir", type=Path, default=None)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=None)
    parser.add_argument("--vqvae-output-dir", type=Path, default=None)
    parser.add_argument("--gpt-output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--native-dry-run", action="store_true")
    parser.add_argument("--vqvae-embedding-dim", type=int, default=64)
    parser.add_argument("--vqvae-n-codes", type=int, default=512)
    parser.add_argument("--vqvae-n-hiddens", type=int, default=64)
    parser.add_argument("--vqvae-n-res-layers", type=int, default=2)
    parser.add_argument("--gpt-hidden-dim", type=int, default=384)
    parser.add_argument("--gpt-heads", type=int, default=4)
    parser.add_argument("--gpt-layers", type=int, default=4)
    parser.add_argument("--gpt-dropout", type=float, default=0.1)
    parser.add_argument("--gpt-attn-dropout", type=float, default=0.1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.context_frames is None:
        args.context_frames = default_context_frames(args.data_track)
    if args.prediction_horizon is None:
        args.prediction_horizon = default_prediction_horizon(args.data_track)
    validate_track_settings(
        args.data_track,
        args.context_frames,
        args.prediction_horizon,
        args.interpolation_method,
    )
    validate_report_horizons(args)
    if args.phase == "train":
        train(args)
    else:
        evaluate(args)


def validate_report_horizons(args: argparse.Namespace) -> None:
    if not args.report_horizons:
        return
    for horizon in args.report_horizons:
        validate_track_settings(args.data_track, args.context_frames, horizon, args.interpolation_method)
        if horizon > args.prediction_horizon:
            raise ValueError(
                f"--report-horizons values must be <= --prediction-horizon: {horizon} > {args.prediction_horizon}"
            )


def train(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        raise ValueError("--output-dir is required for --phase train")
    if not args.train_manifest or not args.val_manifest:
        raise ValueError("--train-manifest and --val-manifest are required for --phase train")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = resolve_manifest(args.dataset_root, args.train_manifest)
    val_manifest = resolve_manifest(args.dataset_root, args.val_manifest)
    train_windows = _load_windows(args, train_manifest)
    val_windows = _load_windows(args, val_manifest)

    if args.frame_predictor == "native_videogpt" and args.mock_model != "copy_last":
        native_checkpoint = train_native_videogpt(args, train_manifest, val_manifest, train_windows, val_windows)
        print(json.dumps(native_checkpoint, indent=2))
        return

    checkpoint_path = args.output_dir / "videogpt_surgwmbench_adapter_checkpoint.json"
    checkpoint = {
        "dataset_name": "SurgWMBench",
        "baseline": "videogpt",
        "model": "VideoGPT",
        "prediction_task": args.prediction_task,
        "data_track": args.data_track,
        "trajectory_target": trajectory_target_for_track(args.data_track),
        "interpolation_method": args.interpolation_method if args.data_track == DENSE_TRACK else None,
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "epochs": args.epochs,
        "seed": args.seed,
        "frame_predictor": args.frame_predictor,
        "trajectory_predictor": args.trajectory_predictor,
        "device": args.device,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
        "trained_native_model": False,
        "note": (
            "Adapter metadata checkpoint. CPU smoke path uses copy-last frames and deterministic "
            "trajectory extrapolation. Native VideoGPT evaluation requires a separately trained "
            "frame-conditioned VideoGPT checkpoint."
        ),
        "timestamp": _timestamp(),
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "num_train_windows": len(train_windows), "num_val_windows": len(val_windows)}, indent=2))


def train_native_videogpt(
    args: argparse.Namespace,
    train_manifest: Path,
    val_manifest: Path,
    train_windows: Sequence[Any],
    val_windows: Sequence[Any],
) -> Dict[str, Any]:
    if args.data_track != SPARSE_TRACK:
        raise ValueError("native_videogpt training currently supports only sparse_20_anchor")
    if args.prediction_task not in {"future_frames", "future_joint"}:
        raise ValueError("native_videogpt training predicts frames; use future_frames or future_joint")
    if args.epochs <= 0 and not args.native_dry_run:
        raise ValueError("native_videogpt training requires --epochs > 0, or use --native-dry-run")

    data_dir = args.native_data_dir or (args.output_dir / "native_data")
    vqvae_output_dir = args.vqvae_output_dir or (args.output_dir / "vqvae")
    gpt_output_dir = args.gpt_output_dir or (args.output_dir / "gpt")
    export_sparse_manifest_videos(
        args.dataset_root,
        train_manifest,
        data_dir / "train" / "surgwmbench",
        args.image_size,
        args.max_clips,
    )
    export_sparse_manifest_videos(
        args.dataset_root,
        val_manifest,
        data_dir / "test" / "surgwmbench",
        args.image_size,
        args.max_clips,
    )

    commands: List[List[str]] = []
    repo_root = Path(__file__).resolve().parents[1]
    vqvae_checkpoint = args.vqvae_checkpoint
    if vqvae_checkpoint is None:
        vqvae_command = [
            sys.executable,
            str(repo_root / "scripts" / "train_vqvae.py"),
            "--data_path",
            str(data_dir),
            "--sequence_length",
            "20",
            "--resolution",
            str(args.image_size),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--gpus",
            str(args.gpus),
            "--max_epochs",
            str(args.epochs),
            "--default_root_dir",
            str(vqvae_output_dir),
            "--embedding_dim",
            str(args.vqvae_embedding_dim),
            "--n_codes",
            str(args.vqvae_n_codes),
            "--n_hiddens",
            str(args.vqvae_n_hiddens),
            "--n_res_layers",
            str(args.vqvae_n_res_layers),
        ]
        commands.append(vqvae_command)
        if not args.native_dry_run:
            run_command(vqvae_command, repo_root)
            vqvae_checkpoint = latest_checkpoint(vqvae_output_dir)
        else:
            vqvae_checkpoint = vqvae_output_dir / "DRY_RUN_VQVAE.ckpt"
    if vqvae_checkpoint is None:
        raise RuntimeError(f"No VQ-VAE checkpoint found under {vqvae_output_dir}; set --vqvae-checkpoint")

    gpt_command = [
        sys.executable,
        str(repo_root / "scripts" / "train_videogpt.py"),
        "--data_path",
        str(data_dir),
        "--sequence_length",
        "20",
        "--resolution",
        str(args.image_size),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--gpus",
        str(args.gpus),
        "--max_epochs",
        str(args.epochs),
        "--default_root_dir",
        str(gpt_output_dir),
        "--vqvae",
        str(vqvae_checkpoint),
        "--n_cond_frames",
        str(args.context_frames),
        "--hidden_dim",
        str(args.gpt_hidden_dim),
        "--heads",
        str(args.gpt_heads),
        "--layers",
        str(args.gpt_layers),
        "--dropout",
        str(args.gpt_dropout),
        "--attn_dropout",
        str(args.gpt_attn_dropout),
    ]
    commands.append(gpt_command)
    gpt_checkpoint = None
    if not args.native_dry_run:
        run_command(gpt_command, repo_root)
        gpt_checkpoint = latest_checkpoint(gpt_output_dir)
    else:
        gpt_checkpoint = gpt_output_dir / "DRY_RUN_VIDEOGPT.ckpt"

    checkpoint_path = args.output_dir / "videogpt_surgwmbench_native_checkpoint.json"
    checkpoint = {
        "dataset_name": "SurgWMBench",
        "baseline": "videogpt",
        "model": "VideoGPT",
        "prediction_task": args.prediction_task,
        "data_track": args.data_track,
        "trajectory_target": trajectory_target_for_track(args.data_track),
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
        "native_data_dir": str(data_dir),
        "vqvae_checkpoint": str(vqvae_checkpoint),
        "gpt_checkpoint": str(gpt_checkpoint) if gpt_checkpoint is not None else None,
        "trained_native_model": not args.native_dry_run,
        "native_dry_run": args.native_dry_run,
        "commands": commands,
        "timestamp": _timestamp(),
    }
    checkpoint_path.write_text(json.dumps(_json_ready(checkpoint), indent=2), encoding="utf-8")
    return {"checkpoint": str(checkpoint_path), **checkpoint}


def evaluate(args: argparse.Namespace) -> None:
    if args.output is None:
        raise ValueError("--output is required for --phase eval")
    if not args.manifest:
        raise ValueError("--manifest is required for --phase eval")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --phase eval")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    manifest = resolve_manifest(args.dataset_root, args.manifest)
    windows = _load_windows(args, manifest)
    artifact_dir = args.output.parent / "sample_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    image_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    weights: List[float] = []
    by_difficulty: Dict[Optional[str], Dict[str, List[Any]]] = {
        "low": {"image": [], "trajectory": [], "weights": []},
        "medium": {"image": [], "trajectory": [], "weights": []},
        "high": {"image": [], "trajectory": [], "weights": []},
        None: {"image": [], "trajectory": [], "weights": []},
    }
    sample_artifacts: List[Dict[str, Any]] = []
    report_horizons = sorted(set(args.report_horizons or []))
    horizon_states = {horizon: new_metric_state() for horizon in report_horizons}

    for window_index, window in enumerate(windows):
        weights.append(window.weight)
        difficulty_key = window.difficulty if window.difficulty in {"low", "medium", "high"} else None
        by_difficulty[difficulty_key]["weights"].append(window.weight)

        pred_coords = None
        pred_frames = None
        if args.prediction_task in {"future_trajectory", "future_joint"}:
            pred_coords = predict_trajectory(window.context_coords, args.prediction_horizon, args.trajectory_predictor)
            row_metrics = trajectory_metrics(pred_coords, window.future_coords)
            trajectory_rows.append(row_metrics)
            by_difficulty[difficulty_key]["trajectory"].append(row_metrics)

        image_row = None
        if args.prediction_task in {"future_frames", "future_joint"}:
            pred_frames = predict_future_frames(args.dataset_root, window.context_frame_paths, args)
            target_frames = load_images(args.dataset_root, window.future_frame_paths, args.image_size)
            image_row = image_sequence_metrics(pred_frames, target_frames)
            image_rows.append(image_row)
            by_difficulty[difficulty_key]["image"].append(image_row)

        for horizon, state in horizon_states.items():
            record_horizon_metrics(
                state=state,
                args=args,
                window=window,
                difficulty_key=difficulty_key,
                horizon=horizon,
                pred_coords=pred_coords,
                pred_frames=pred_frames,
            )

        if len(sample_artifacts) < args.max_sample_artifacts:
            sample_artifacts.append(
                write_sample_artifact(
                    artifact_dir,
                    len(sample_artifacts),
                    window,
                    pred_coords,
                    pred_frames,
                )
            )

    result = {
        "dataset_name": "SurgWMBench",
        "baseline": "videogpt",
        "model": "VideoGPT",
        "manifest": str(manifest),
        "prediction_task": args.prediction_task,
        "data_track": args.data_track,
        "trajectory_target": trajectory_target_for_track(args.data_track),
        "interpolation_method": args.interpolation_method if args.data_track == DENSE_TRACK else None,
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "report_horizons": report_horizons,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "image_metrics_overall": aggregate_metric_dicts(image_rows, weights) if image_rows else empty_image_metrics(),
        "trajectory_metrics_overall": aggregate_metric_dicts(trajectory_rows, weights) if trajectory_rows else empty_trajectory_metrics(),
        "metrics_by_difficulty": build_difficulty_metrics(by_difficulty),
        "sample_artifacts": sample_artifacts,
        "num_windows": len(windows),
        "timestamp": _timestamp(),
    }
    if horizon_states:
        result["horizon_metrics"] = {
            str(horizon): summarize_metric_state(horizon, state)
            for horizon, state in horizon_states.items()
        }
    args.output.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_windows": len(windows)}, indent=2))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_windows(args: argparse.Namespace, manifest: Path):
    return sample_windows(
        dataset_root=args.dataset_root,
        manifest=manifest,
        data_track=args.data_track,
        context_frames=args.context_frames,
        prediction_horizon=args.prediction_horizon,
        interpolation_method=args.interpolation_method,
        max_clips=args.max_clips,
        max_windows_per_clip=args.max_windows,
        window_stride=args.window_stride,
        inverse_window_count_reweight=args.inverse_window_count_reweight,
    )


def predict_trajectory(
    context_coords: Sequence[Sequence[float]],
    horizon: int,
    predictor: str,
) -> List[List[float]]:
    context = np.asarray(context_coords, dtype=np.float64)
    if predictor == "copy_last" or len(context) < 2:
        step = np.zeros(2, dtype=np.float64)
    else:
        step = context[-1] - context[-2]
    start = context[-1]
    predictions = [np.clip(start + step * (idx + 1), 0.0, 1.0).tolist() for idx in range(horizon)]
    return [[float(item[0]), float(item[1])] for item in predictions]


def predict_future_frames(dataset_root: Path, context_frame_paths: Sequence[str], args: argparse.Namespace) -> List[np.ndarray]:
    if args.frame_predictor == "native_videogpt" and args.mock_model != "copy_last":
        return sample_native_videogpt_future_frames(dataset_root, context_frame_paths, args)
    last_context_frame = load_image(dataset_root / context_frame_paths[-1], args.image_size)
    return [last_context_frame.copy() for _ in range(args.prediction_horizon)]


def record_horizon_metrics(
    state: Dict[str, Any],
    args: argparse.Namespace,
    window: Any,
    difficulty_key: Optional[str],
    horizon: int,
    pred_coords: Optional[Sequence[Sequence[float]]],
    pred_frames: Optional[Sequence[np.ndarray]],
) -> None:
    state["weights"].append(window.weight)
    state["by_difficulty"][difficulty_key]["weights"].append(window.weight)
    if args.prediction_task in {"future_trajectory", "future_joint"} and pred_coords is not None:
        trajectory_row = trajectory_metrics(pred_coords[:horizon], window.future_coords[:horizon])
        state["trajectory"].append(trajectory_row)
        state["by_difficulty"][difficulty_key]["trajectory"].append(trajectory_row)
    if args.prediction_task in {"future_frames", "future_joint"} and pred_frames is not None:
        target_frames = load_images(args.dataset_root, window.future_frame_paths[:horizon], args.image_size)
        image_row = image_sequence_metrics(pred_frames[:horizon], target_frames)
        state["image"].append(image_row)
        state["by_difficulty"][difficulty_key]["image"].append(image_row)


def new_metric_state() -> Dict[str, Any]:
    return {
        "image": [],
        "trajectory": [],
        "weights": [],
        "by_difficulty": {
            "low": {"image": [], "trajectory": [], "weights": []},
            "medium": {"image": [], "trajectory": [], "weights": []},
            "high": {"image": [], "trajectory": [], "weights": []},
            None: {"image": [], "trajectory": [], "weights": []},
        },
    }


def summarize_metric_state(horizon: int, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_horizon": horizon,
        "image_metrics_overall": aggregate_metric_dicts(state["image"], state["weights"]) if state["image"] else empty_image_metrics(),
        "trajectory_metrics_overall": aggregate_metric_dicts(state["trajectory"], state["weights"]) if state["trajectory"] else empty_trajectory_metrics(),
        "metrics_by_difficulty": build_difficulty_metrics(state["by_difficulty"]),
        "num_windows": len(state["weights"]),
    }


def load_images(dataset_root: Path, frame_paths: Sequence[str], image_size: int) -> List[np.ndarray]:
    return [load_image(dataset_root / frame_path, image_size) for frame_path in frame_paths]


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        return np.asarray(image, dtype=np.uint8)


def sample_native_videogpt_future_frames(
    dataset_root: Path,
    context_frame_paths: Sequence[str],
    args: argparse.Namespace,
) -> List[np.ndarray]:
    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix == ".json":
        raise RuntimeError(
            "native_videogpt requires a real VideoGPT Lightning checkpoint, not the adapter "
            f"metadata checkpoint: {checkpoint}"
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"native_videogpt checkpoint does not exist: {checkpoint}")

    try:
        import torch

        from .gpt import VideoGPT
    except Exception as exc:  # pragma: no cover - depends on the legacy VideoGPT environment.
        raise RuntimeError(
            "native_videogpt requires the legacy VideoGPT runtime dependencies, including "
            "torch and pytorch_lightning."
        ) from exc

    model = getattr(args, "_native_videogpt_model", None)
    if model is None:
        model = VideoGPT.load_from_checkpoint(str(checkpoint), map_location=args.device)
        model = model.to(args.device)
        model.eval()
        if getattr(model.args, "class_cond", False):
            raise RuntimeError("native_videogpt SurgWMBench adapter does not support class-conditioned checkpoints")
        if not getattr(model, "use_frame_cond", False):
            raise RuntimeError("native_videogpt future prediction requires a frame-conditioned VideoGPT checkpoint")
        if int(getattr(model.args, "n_cond_frames", -1)) != int(args.context_frames):
            raise RuntimeError(
                "VideoGPT checkpoint n_cond_frames does not match --context-frames: "
                f"{getattr(model.args, 'n_cond_frames', None)} vs {args.context_frames}"
            )
        setattr(args, "_native_videogpt_model", model)

    resolution = int(getattr(model.args, "resolution", args.image_size))
    context = []
    for frame_path in context_frame_paths[: args.context_frames]:
        context.append(load_image(dataset_root / frame_path, resolution))
    video = np.stack(context, axis=0)
    video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()
    video_tensor = (video_tensor / 255.0) - 0.5
    batch = {"video": video_tensor.to(args.device)}

    with torch.no_grad():
        samples = model.sample(1, batch)
    sample = samples[0].detach().cpu()  # C, T, H, W in [0, 1]
    if sample.ndim != 4:
        raise RuntimeError(f"native_videogpt returned an unexpected sample shape: {tuple(sample.shape)}")
    available_future = int(sample.shape[1]) - int(args.context_frames)
    if available_future < int(args.prediction_horizon):
        raise RuntimeError(
            "native_videogpt checkpoint generated too few future frames for this horizon: "
            f"available {available_future}, requested {args.prediction_horizon}"
        )
    frames: List[np.ndarray] = []
    for offset in range(args.prediction_horizon):
        frame_index = int(args.context_frames) + offset
        frame = sample[:, frame_index].permute(1, 2, 0).numpy()
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame_uint8.shape[:2] != (args.image_size, args.image_size):
            frame_uint8 = np.asarray(Image.fromarray(frame_uint8).resize((args.image_size, args.image_size)))
        frames.append(frame_uint8)
    return frames


def image_sequence_metrics(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> Dict[str, Optional[float]]:
    if len(predictions) != len(targets):
        raise ValueError(
            f"Prediction and target frame sequence lengths differ: {len(predictions)} vs {len(targets)}"
        )
    if not predictions:
        return empty_image_metrics()
    rows = [image_metrics(prediction, target) for prediction, target in zip(predictions, targets)]
    return aggregate_metric_dicts(rows, [1.0] * len(rows))


def write_sample_artifact(
    artifact_dir: Path,
    index: int,
    window: Any,
    pred_coords: Optional[List[List[float]]],
    pred_frames: Optional[Sequence[np.ndarray]],
) -> Dict[str, Any]:
    payload = {
        "clip_id": window.clip_id,
        "difficulty": window.difficulty,
        "context_indices": window.context_indices,
        "future_indices": window.future_indices,
        "context_coords": window.context_coords,
        "future_coords": window.future_coords,
        "predicted_coords": pred_coords,
    }
    json_path = artifact_dir / f"sample_{index:03d}.json"
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    artifact: Dict[str, Any] = {"metadata": str(json_path)}
    if pred_frames is not None:
        frames_dir = artifact_dir / f"sample_{index:03d}_pred_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: List[str] = []
        for frame_index, pred_frame in enumerate(pred_frames):
            frame_path = frames_dir / f"{frame_index:03d}.png"
            Image.fromarray(pred_frame).save(frame_path)
            frame_paths.append(str(frame_path))
        artifact["pred_frames"] = frame_paths
    return artifact


def export_sparse_manifest_videos(
    dataset_root: Path,
    manifest: Path,
    split_dir: Path,
    image_size: int,
    max_clips: Optional[int],
) -> None:
    import imageio.v2 as imageio

    split_dir.mkdir(parents=True, exist_ok=True)
    for row in read_jsonl(manifest, max_clips=max_clips):
        annotation = load_annotation(dataset_root, row)
        anchors = sorted(annotation.get("human_anchors", []), key=lambda item: int(item["anchor_idx"]))
        if len(anchors) != 20:
            raise ValueError(f"{row.get('patient_id')}/{row.get('trajectory_id')} expected exactly 20 human anchors")
        frame_path_by_index = {
            int(frame["local_frame_idx"]): str(frame["frame_path"])
            for frame in annotation.get("frames", [])
        }
        output_path = split_dir / f"{safe_name(row.get('patient_id'), row.get('trajectory_id'))}.mp4"
        frame_paths = [frame_path_by_index[int(anchor["local_frame_idx"])] for anchor in anchors]
        frames = [load_image(dataset_root / frame_path, image_size) for frame_path in frame_paths]
        imageio.mimsave(output_path, frames, fps=8, macro_block_size=1)


def safe_name(patient_id: Any, trajectory_id: Any) -> str:
    raw = f"{patient_id}_{trajectory_id}"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw)


def run_command(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("**/*.ckpt"), key=lambda item: item.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint found under {output_dir}")
    return checkpoints[-1]


def build_difficulty_metrics(grouped: Dict[Optional[str], Dict[str, List[Any]]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key in ("low", "medium", "high", None):
        label = "null" if key is None else key
        group = grouped[key]
        output[label] = {
            "num_windows": len(group["weights"]),
            "image_metrics_overall": aggregate_metric_dicts(group["image"], group["weights"]) if group["image"] else empty_image_metrics(),
            "trajectory_metrics_overall": aggregate_metric_dicts(group["trajectory"], group["weights"]) if group["trajectory"] else empty_trajectory_metrics(),
        }
    return output


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    main()
