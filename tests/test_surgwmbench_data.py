import json

import imageio.v2 as imageio
import numpy as np
import torch

from videogpt.surgwmbench_data import (
    SurgWMBenchAnchorDataset,
    letterbox_frame,
    restore_letterboxed_frame,
    surgwmbench_collate,
)
from videogpt.surgwmbench_metrics import compute_psnr, compute_ssim


def _write_toy_surgwmbench(root):
    clip_dir = root / "clips" / "video_01" / "traj_01"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    (root / "manifests").mkdir()

    frames = []
    anchors = []
    sampled_indices = list(range(20))
    for idx in sampled_indices:
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[..., 0] = idx
        image[..., 1] = 255 - idx
        frame_rel = f"clips/video_01/traj_01/frames/{idx:06d}.png"
        imageio.imwrite(root / frame_rel, image)
        frames.append({
            "local_frame_idx": idx,
            "source_frame_idx": idx + 100,
            "frame_path": frame_rel,
            "is_human_labeled": True,
            "anchor_idx": idx,
            "human_coord_px": [float(idx), float(idx + 1)],
            "human_coord_norm": [0.1, 0.2],
            "coord_source": "human",
        })
        anchors.append({
            "anchor_idx": idx,
            "old_frame_idx": idx,
            "local_frame_idx": idx,
            "source_frame_idx": idx + 100,
            "label_name": f"Label {idx + 1}",
            "value": idx + 1,
            "coord_px": [float(idx), float(idx + 1)],
            "coord_norm": [0.1, 0.2],
        })

    annotation = {
        "dataset_version": "SurgWMBench",
        "patient_id": "video_01",
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "traj_01",
        "difficulty": "low",
        "num_frames": 20,
        "image_size": {"width": 20, "height": 10},
        "coordinate_format": "pixel_xy",
        "coordinate_origin": "top_left",
        "num_human_anchors": 20,
        "sampled_indices": sampled_indices,
        "available_interpolation_methods": ["linear"],
        "default_interpolation_method": "linear",
        "frames": frames,
        "human_anchors": anchors,
        "interpolation_files": {"linear": "interpolations/video_01/traj_01.linear.json"},
    }
    annotation_path = clip_dir / "annotation.json"
    annotation_path.write_text(json.dumps(annotation))

    row = {
        "annotation_path": "clips/video_01/traj_01/annotation.json",
        "dataset_version": "SurgWMBench",
        "default_interpolation_method": "linear",
        "difficulty": "low",
        "frames_dir": "clips/video_01/traj_01/frames",
        "interpolation_files": {"linear": "interpolations/video_01/traj_01.linear.json"},
        "num_frames": 20,
        "num_human_anchors": 20,
        "patient_id": "video_01",
        "sampled_indices": sampled_indices,
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "traj_01",
    }
    manifest = root / "manifests" / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest


def test_surgwmbench_anchor_dataset_loads_20_anchor_frames(tmp_path):
    manifest = _write_toy_surgwmbench(tmp_path)
    dataset = SurgWMBenchAnchorDataset(tmp_path, manifest, resolution=16)

    sample = dataset[0]

    assert sample["video"].shape == (3, 20, 16, 16)
    assert sample["anchor_local_frame_indices"].tolist() == list(range(20))
    assert len(sample["frame_paths"]) == 20
    assert sample["difficulty"] == "low"


def test_surgwmbench_collate_preserves_metadata(tmp_path):
    manifest = _write_toy_surgwmbench(tmp_path)
    dataset = SurgWMBenchAnchorDataset(tmp_path, manifest, resolution=16)

    batch = surgwmbench_collate([dataset[0], dataset[0]])

    assert batch["video"].shape == (2, 3, 20, 16, 16)
    assert batch["patient_id"] == ["video_01", "video_01"]
    assert len(batch["frame_paths"][0]) == 20


def test_letterbox_restore_returns_original_size():
    frame = torch.rand(3, 10, 20)
    letterboxed, geometry = letterbox_frame(frame, resolution=16)

    restored = restore_letterboxed_frame(letterboxed, geometry)

    assert letterboxed.shape == (3, 16, 16)
    assert restored.shape == (3, 10, 20)


def test_basic_image_metrics_identical_images():
    image = torch.ones(3, 12, 12)

    assert compute_psnr(image, image) == float("inf")
    assert compute_ssim(image, image) == 1.0
