from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


SPARSE_TRACK = "sparse_20_anchor"
DENSE_TRACK = "dense_pseudo"
SPARSE_TARGET = "sparse_human_anchors"
DENSE_TARGET = "pseudo_coordinates"
INTERPOLATION_METHODS = ("linear", "pchip", "akima", "cubic_spline")


@dataclass(frozen=True)
class SurgWMBenchWindow:
    clip_id: str
    patient_id: str
    trajectory_id: str
    difficulty: Optional[str]
    data_track: str
    trajectory_target: str
    interpolation_method: str
    context_indices: List[int]
    future_indices: List[int]
    context_coords: List[List[float]]
    future_coords: List[List[float]]
    context_frame_paths: List[str]
    future_frame_paths: List[str]
    weight: float = 1.0


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_jsonl(path: Path, max_clips: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_number} is not a JSON object: {path}")
            rows.append(row)
            if max_clips is not None and len(rows) >= max_clips:
                break
    return rows


def resolve_dataset_path(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def resolve_manifest(dataset_root: Path, manifest: str | Path) -> Path:
    return resolve_dataset_path(dataset_root, manifest)


def load_annotation(dataset_root: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    return read_json(resolve_dataset_path(dataset_root, row["annotation_path"]))


def load_interpolation(
    dataset_root: Path,
    row: Dict[str, Any],
    interpolation_method: str,
) -> Dict[str, Any]:
    files = row.get("interpolation_files") or {}
    if interpolation_method not in files:
        available = ", ".join(sorted(files))
        raise ValueError(
            f"Interpolation method {interpolation_method!r} is not available for "
            f"{row.get('patient_id')}/{row.get('trajectory_id')}. Available: {available}"
        )
    return read_json(resolve_dataset_path(dataset_root, files[interpolation_method]))


def trajectory_target_for_track(data_track: str) -> str:
    if data_track == SPARSE_TRACK:
        return SPARSE_TARGET
    if data_track == DENSE_TRACK:
        return DENSE_TARGET
    raise ValueError(f"Unknown data track: {data_track}")


def validate_track_settings(
    data_track: str,
    context_frames: int,
    prediction_horizon: int,
    interpolation_method: str,
) -> None:
    if data_track == SPARSE_TRACK:
        if context_frames != 5:
            raise ValueError("sparse_20_anchor requires --context-frames 5")
        if prediction_horizon not in {5, 10, 15}:
            raise ValueError("sparse_20_anchor supports --prediction-horizon 5, 10, or 15")
    elif data_track == DENSE_TRACK:
        if context_frames != 8:
            raise ValueError("dense_pseudo requires --context-frames 8")
        if prediction_horizon != 16:
            raise ValueError("dense_pseudo requires --prediction-horizon 16")
        if interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(
                "dense_pseudo supports interpolation methods: "
                + ", ".join(INTERPOLATION_METHODS)
            )
    else:
        raise ValueError(f"Unknown data track: {data_track}")


def default_context_frames(data_track: str) -> int:
    return 5 if data_track == SPARSE_TRACK else 8


def default_prediction_horizon(data_track: str) -> int:
    return 5 if data_track == SPARSE_TRACK else 16


def _clip_id(row: Dict[str, Any]) -> str:
    return f"{row.get('patient_id')}/{row.get('trajectory_id')}"


def _frame_path_by_index(annotation: Dict[str, Any]) -> Dict[int, str]:
    return {
        int(frame["local_frame_idx"]): str(frame["frame_path"])
        for frame in annotation.get("frames", [])
    }


def _coord_norm(item: Dict[str, Any]) -> List[float]:
    coord = item.get("coord_norm") or item.get("human_coord_norm")
    if coord is None:
        raise ValueError(f"Missing normalized coordinate in item: {item}")
    return [float(coord[0]), float(coord[1])]


def iter_sparse_windows(
    dataset_root: Path,
    rows: Iterable[Dict[str, Any]],
    context_frames: int,
    prediction_horizon: int,
    max_windows_per_clip: Optional[int] = None,
) -> Iterator[SurgWMBenchWindow]:
    for row in rows:
        if max_windows_per_clip == 0:
            continue
        annotation = load_annotation(dataset_root, row)
        anchors = sorted(annotation.get("human_anchors", []), key=lambda item: item["anchor_idx"])
        if len(anchors) != 20:
            raise ValueError(f"{_clip_id(row)} expected exactly 20 human anchors")
        total = context_frames + prediction_horizon
        if total > len(anchors):
            continue
        frame_paths = _frame_path_by_index(annotation)
        context = anchors[:context_frames]
        future = anchors[context_frames:total]
        window = SurgWMBenchWindow(
            clip_id=_clip_id(row),
            patient_id=str(row["patient_id"]),
            trajectory_id=str(row["trajectory_id"]),
            difficulty=row.get("difficulty"),
            data_track=SPARSE_TRACK,
            trajectory_target=SPARSE_TARGET,
            interpolation_method="none",
            context_indices=[int(item["local_frame_idx"]) for item in context],
            future_indices=[int(item["local_frame_idx"]) for item in future],
            context_coords=[_coord_norm(item) for item in context],
            future_coords=[_coord_norm(item) for item in future],
            context_frame_paths=[frame_paths[int(item["local_frame_idx"])] for item in context],
            future_frame_paths=[frame_paths[int(item["local_frame_idx"])] for item in future],
            weight=1.0,
        )
        yield window


def iter_dense_windows(
    dataset_root: Path,
    rows: Iterable[Dict[str, Any]],
    context_frames: int,
    prediction_horizon: int,
    interpolation_method: str,
    max_windows_per_clip: Optional[int] = None,
    window_stride: int = 1,
    inverse_window_count_reweight: bool = False,
) -> Iterator[SurgWMBenchWindow]:
    total = context_frames + prediction_horizon
    for row in rows:
        annotation = load_annotation(dataset_root, row)
        interpolation = load_interpolation(dataset_root, row, interpolation_method)
        coords = sorted(
            interpolation.get("coordinates", []),
            key=lambda item: int(item["local_frame_idx"]),
        )
        frame_paths = _frame_path_by_index(annotation)
        if len(coords) < total:
            continue
        starts = list(range(0, len(coords) - total + 1, max(1, window_stride)))
        if max_windows_per_clip is not None:
            starts = starts[:max_windows_per_clip]
        if not starts:
            continue
        weight = 1.0 / len(starts) if inverse_window_count_reweight else 1.0
        for start in starts:
            context = coords[start : start + context_frames]
            future = coords[start + context_frames : start + total]
            yield SurgWMBenchWindow(
                clip_id=_clip_id(row),
                patient_id=str(row["patient_id"]),
                trajectory_id=str(row["trajectory_id"]),
                difficulty=row.get("difficulty"),
                data_track=DENSE_TRACK,
                trajectory_target=DENSE_TARGET,
                interpolation_method=interpolation_method,
                context_indices=[int(item["local_frame_idx"]) for item in context],
                future_indices=[int(item["local_frame_idx"]) for item in future],
                context_coords=[_coord_norm(item) for item in context],
                future_coords=[_coord_norm(item) for item in future],
                context_frame_paths=[frame_paths[int(item["local_frame_idx"])] for item in context],
                future_frame_paths=[frame_paths[int(item["local_frame_idx"])] for item in future],
                weight=weight,
            )


def sample_windows(
    dataset_root: Path,
    manifest: Path,
    data_track: str,
    context_frames: int,
    prediction_horizon: int,
    interpolation_method: str = "linear",
    max_clips: Optional[int] = None,
    max_windows_per_clip: Optional[int] = None,
    window_stride: int = 1,
    inverse_window_count_reweight: bool = False,
) -> List[SurgWMBenchWindow]:
    rows = read_jsonl(manifest, max_clips=max_clips)
    if data_track == SPARSE_TRACK:
        return list(
            iter_sparse_windows(
                dataset_root,
                rows,
                context_frames,
                prediction_horizon,
                max_windows_per_clip=max_windows_per_clip,
            )
        )
    if data_track == DENSE_TRACK:
        return list(
            iter_dense_windows(
                dataset_root,
                rows,
                context_frames,
                prediction_horizon,
                interpolation_method,
                max_windows_per_clip=max_windows_per_clip,
                window_stride=window_stride,
                inverse_window_count_reweight=inverse_window_count_reweight,
            )
        )
    raise ValueError(f"Unknown data track: {data_track}")
