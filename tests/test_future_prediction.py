from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from videogpt.future_prediction import main
from videogpt.surgwmbench_data import DENSE_TARGET, SPARSE_TARGET, sample_windows


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset(root: Path, num_frames: int = 40) -> Path:
    clip_dir = root / "clips" / "video_01" / "traj_01"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_frames):
        color = (idx % 255, (idx * 3) % 255, (idx * 7) % 255)
        Image.new("RGB", (32, 24), color=color).save(frames_dir / f"{idx:06d}.png")

    sampled_indices = [round(i * (num_frames - 1) / 19) for i in range(20)]
    human_anchors = []
    frames = []
    sampled_to_anchor = {frame_idx: anchor_idx for anchor_idx, frame_idx in enumerate(sampled_indices)}
    for idx in range(num_frames):
        anchor_idx = sampled_to_anchor.get(idx)
        coord = [idx / max(1, num_frames - 1), 1.0 - idx / max(1, num_frames - 1)]
        frames.append(
            {
                "local_frame_idx": idx,
                "source_frame_idx": idx + 100,
                "frame_path": f"clips/video_01/traj_01/frames/{idx:06d}.png",
                "is_human_labeled": anchor_idx is not None,
                "anchor_idx": anchor_idx,
                "human_coord_px": [coord[0] * 32, coord[1] * 24] if anchor_idx is not None else None,
                "human_coord_norm": coord if anchor_idx is not None else None,
                "coord_source": "human" if anchor_idx is not None else "unlabeled",
            }
        )
        if anchor_idx is not None:
            human_anchors.append(
                {
                    "anchor_idx": anchor_idx,
                    "old_frame_idx": anchor_idx,
                    "local_frame_idx": idx,
                    "source_frame_idx": idx + 100,
                    "label_name": "Label 1",
                    "value": 1,
                    "coord_px": [coord[0] * 32, coord[1] * 24],
                    "coord_norm": coord,
                }
            )

    interpolation_files = {
        method: f"interpolations/video_01/traj_01.{method}.json"
        for method in ("linear", "pchip", "akima", "cubic_spline")
    }
    annotation = {
        "dataset_version": "SurgWMBench",
        "patient_id": "video_01",
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "traj_01",
        "difficulty": "low",
        "num_frames": num_frames,
        "image_size": {"width": 32, "height": 24},
        "coordinate_format": "pixel_xy",
        "coordinate_origin": "top_left",
        "num_human_anchors": 20,
        "sampled_indices": sampled_indices,
        "available_interpolation_methods": list(interpolation_files),
        "default_interpolation_method": "linear",
        "frames": frames,
        "human_anchors": human_anchors,
        "interpolation_files": interpolation_files,
    }
    _write_json(clip_dir / "annotation.json", annotation)

    coordinates = []
    for idx in range(num_frames):
        anchor_idx = sampled_to_anchor.get(idx)
        coord = [idx / max(1, num_frames - 1), 1.0 - idx / max(1, num_frames - 1)]
        coordinates.append(
            {
                "local_frame_idx": idx,
                "coord_px": [coord[0] * 32, coord[1] * 24],
                "coord_norm": coord,
                "source": "human" if anchor_idx is not None else "interpolated",
                "anchor_idx": anchor_idx,
                "confidence": 1.0 if anchor_idx is not None else 0.6,
                "label_weight": 1.0 if anchor_idx is not None else 0.5,
                "is_out_of_bounds": False,
            }
        )
    for method, rel_path in interpolation_files.items():
        _write_json(
            root / rel_path,
            {
                "dataset_version": "SurgWMBench",
                "patient_id": "video_01",
                "trajectory_id": "traj_01",
                "interpolation_method": method,
                "num_frames": num_frames,
                "image_size": {"width": 32, "height": 24},
                "coordinates": coordinates,
            },
        )

    row = {
        "annotation_path": "clips/video_01/traj_01/annotation.json",
        "dataset_version": "SurgWMBench",
        "default_interpolation_method": "linear",
        "difficulty": "low",
        "frames_dir": "clips/video_01/traj_01/frames",
        "interpolation_files": interpolation_files,
        "num_frames": num_frames,
        "num_human_anchors": 20,
        "patient_id": "video_01",
        "sampled_indices": sampled_indices,
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "traj_01",
    }
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (manifests / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def test_sparse_sampler_uses_human_anchor_windows(tmp_path: Path):
    root = _make_dataset(tmp_path)
    windows = sample_windows(
        root,
        root / "manifests" / "test.jsonl",
        data_track="sparse_20_anchor",
        context_frames=5,
        prediction_horizon=10,
    )
    assert len(windows) == 1
    assert windows[0].trajectory_target == SPARSE_TARGET
    assert len(windows[0].context_coords) == 5
    assert len(windows[0].future_coords) == 10
    assert windows[0].context_indices == [0, 2, 4, 6, 8]


def test_dense_sampler_uses_pseudo_coordinates_and_window_weights(tmp_path: Path):
    root = _make_dataset(tmp_path)
    windows = sample_windows(
        root,
        root / "manifests" / "test.jsonl",
        data_track="dense_pseudo",
        context_frames=8,
        prediction_horizon=16,
        interpolation_method="pchip",
        max_windows_per_clip=2,
        inverse_window_count_reweight=True,
    )
    assert len(windows) == 2
    assert windows[0].trajectory_target == DENSE_TARGET
    assert windows[0].interpolation_method == "pchip"
    assert windows[0].weight == 0.5
    assert len(windows[0].future_coords) == 16


def test_train_and_eval_cli_write_schema_compatible_outputs(tmp_path: Path):
    root = _make_dataset(tmp_path)
    output_dir = tmp_path / "run"
    main(
        [
            "--phase",
            "train",
            "--prediction-task",
            "future_joint",
            "--data-track",
            "dense_pseudo",
            "--dataset-root",
            str(root),
            "--train-manifest",
            "manifests/train.jsonl",
            "--val-manifest",
            "manifests/val.jsonl",
            "--context-frames",
            "8",
            "--prediction-horizon",
            "16",
            "--max-clips",
            "1",
            "--max-windows",
            "1",
            "--epochs",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    )
    checkpoint = output_dir / "videogpt_surgwmbench_adapter_checkpoint.json"
    assert checkpoint.exists()

    metrics_path = output_dir / "metrics.json"
    main(
        [
            "--phase",
            "eval",
            "--prediction-task",
            "future_joint",
            "--data-track",
            "dense_pseudo",
            "--dataset-root",
            str(root),
            "--manifest",
            "manifests/test.jsonl",
            "--checkpoint",
            str(checkpoint),
            "--context-frames",
            "8",
            "--prediction-horizon",
            "16",
            "--max-clips",
            "1",
            "--max-windows",
            "1",
            "--mock-model",
            "copy_last",
            "--max-sample-artifacts",
            "1",
            "--output",
            str(metrics_path),
        ]
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "SurgWMBench"
    assert payload["baseline"] == "videogpt"
    assert payload["trajectory_target"] == DENSE_TARGET
    assert payload["num_windows"] == 1
    assert set(payload["trajectory_metrics_overall"]) >= {"ade", "fde", "hausdorff"}
    assert set(payload["image_metrics_overall"]) >= {"mse", "psnr", "ssim", "lpips", "fvd"}
    assert set(payload["metrics_by_difficulty"]) == {"low", "medium", "high", "null"}
    assert len(payload["sample_artifacts"]) == 1


def test_native_videogpt_requires_real_checkpoint(tmp_path: Path):
    root = _make_dataset(tmp_path)
    checkpoint = tmp_path / "adapter_checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="real VideoGPT Lightning checkpoint"):
        main(
            [
                "--phase",
                "eval",
                "--prediction-task",
                "future_frames",
                "--data-track",
                "sparse_20_anchor",
                "--dataset-root",
                str(root),
                "--manifest",
                "manifests/test.jsonl",
                "--checkpoint",
                str(checkpoint),
                "--context-frames",
                "5",
                "--prediction-horizon",
                "5",
                "--max-clips",
                "1",
                "--max-windows",
                "1",
                "--frame-predictor",
                "native_videogpt",
                "--output",
                str(tmp_path / "metrics.json"),
            ]
        )
