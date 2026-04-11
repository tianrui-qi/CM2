from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import yaml
import zarr
from PIL import Image
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from scipy import ndimage as ndi
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_SCHEMA_PATH = REPO_ROOT / "config" / "schema" / "save.yaml"

TRACE_SOURCE_KEY = "c"
TRACE_FILE_NAME = "traces_c.float32.bin"
POINTS_FILE_NAME = "points.json"
METADATA_FILE_NAME = "metadata.json"
BACKGROUND_DIRNAME = "backgrounds"

PATCH_NAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)$")

QUALITY_PROFILE_KEYS = (
    "r_value",
    "snr",
    "bl",
    "lam",
    "neurons_sn",
    "g_0",
    "g_1",
    "t_peak",
    "t_half",
    "r_value_unreliable_joint_only",
)


@dataclass
class PatchMeta:
    patch_name: str
    qrow: int
    qcol: int
    prow: int
    pcol: int
    h: int
    w: int
    t: int
    movie_path: Path
    model_path: Path
    y0: int = 0
    y1: int = 0
    x0: int = 0
    x1: int = 0
    core_h: int = 0
    core_w: int = 0
    core_ly0: int = 0
    core_ly1: int = 0
    core_lx0: int = 0
    core_lx1: int = 0


@dataclass(frozen=True)
class CropPatchSpec:
    name: str
    quadrant_row: int
    quadrant_col: int
    patch_row: int
    patch_col: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class QcThresholdSpec:
    key: str
    scale: str
    mean: float
    std: float
    lower_z: float | None
    upper_z: float | None


@contextmanager
def silence_stdio():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def _load_crop_params(crop_load_fold: Path) -> dict[str, object]:
    crop_load_fold = Path(crop_load_fold)
    if not crop_load_fold.is_dir():
        raise NotADirectoryError(
            f"crop_load_fold must be a crop folder containing para.json: {crop_load_fold}"
        )
    para_path = crop_load_fold / "para.json"
    if not para_path.is_file():
        raise FileNotFoundError(f"Crop para.json not found: {para_path}")
    payload = json.loads(para_path.read_text(encoding="utf-8"))
    patch_specs_raw = payload.get("patch_specs")
    if not isinstance(patch_specs_raw, list) or not patch_specs_raw:
        raise ValueError(f"Invalid crop para.json patch_specs: {para_path}")
    patch_specs = [
        CropPatchSpec(
            name=str(spec["name"]),
            quadrant_row=int(spec["quadrant_row"]),
            quadrant_col=int(spec["quadrant_col"]),
            patch_row=int(spec["patch_row"]),
            patch_col=int(spec["patch_col"]),
            y0=int(spec["y0"]),
            y1=int(spec["y1"]),
            x0=int(spec["x0"]),
            x1=int(spec["x1"]),
        )
        for spec in patch_specs_raw
    ]
    payload["patch_specs"] = patch_specs
    return payload


def _parse_patch_name(stem: str) -> tuple[int, int, int, int]:
    match = PATCH_NAME_RE.match(stem)
    if match is None:
        raise ValueError(
            f"Patch name must match '<qrow>-<qcol>-<prow>-<pcol>', got: {stem}"
        )
    return tuple(int(x) for x in match.groups())  # type: ignore[return-value]


def _model_file_to_patch_name(path: Path) -> str:
    name = path.name
    if name.endswith(".tif.hdf5"):
        return name[: -len(".tif.hdf5")]
    if name.endswith(".hdf5"):
        return path.stem
    raise ValueError(f"Unsupported model filename: {name}")


def get_model_dir(extract_load_fold: Path) -> Path:
    return extract_load_fold / "patch-model"


def get_profile_dir(extract_load_fold: Path) -> Path:
    return extract_load_fold / "patch-profile"


def _collect_patch_meta(
    y_load_fold: Path,
    model_dir: Path,
) -> list[PatchMeta]:
    patches: list[PatchMeta] = []
    model_files = sorted(model_dir.glob("*.hdf5"), key=lambda p: p.name)
    if not model_files:
        raise FileNotFoundError(f"No .hdf5 model files found in: {model_dir}")

    for model_path in model_files:
        patch_name = _model_file_to_patch_name(model_path)
        qrow, qcol, prow, pcol = _parse_patch_name(patch_name)
        movie_path = y_load_fold / f"{patch_name}.tif"
        if not movie_path.is_file():
            raise FileNotFoundError(
                f"Missing patch movie for model {model_path.name}: {movie_path}"
            )
        with tifffile.TiffFile(movie_path) as tif:
            video = zarr.open(tif.aszarr(), mode="r")
            if video.ndim != 3:
                raise ValueError(
                    f"Patch movie must be TYX, got ndim={video.ndim}: {movie_path}"
                )
            t = int(video.shape[0])
            h = int(video.shape[1])
            w = int(video.shape[2])
        patches.append(
            PatchMeta(
                patch_name=patch_name,
                qrow=qrow,
                qcol=qcol,
                prow=prow,
                pcol=pcol,
                h=h,
                w=w,
                t=t,
                movie_path=movie_path,
                model_path=model_path,
            )
        )
    return patches


def _metric_to_qc_space(raw_value: float, scale: str) -> float:
    value = float(raw_value)
    if scale == "linear":
        return value
    if scale == "log":
        if np.isfinite(value) and value > 0.0:
            return float(np.log10(value))
        return float("-inf")
    raise ValueError(f"Unsupported QC metric scale: {scale}")


def _normalize_qc_bounds(
    bounds: Sequence[float | None] | None,
    key: str,
) -> tuple[float | None, float | None]:
    if bounds is None:
        return None, None
    if isinstance(bounds, (str, bytes)):
        raise TypeError(
            f"QC threshold for {key} must be [lower, upper] or null, got string-like value."
        )
    bounds_list = list(bounds)
    if len(bounds_list) != 2:
        raise ValueError(
            f"QC threshold for {key} must have exactly two entries: [lower, upper]."
        )

    def _cast_one(value: float | None) -> float | None:
        if value is None:
            return None
        return float(value)

    return _cast_one(bounds_list[0]), _cast_one(bounds_list[1])


def _load_qc_threshold_specs(
    extract_load_fold: Path,
    t_snr: Sequence[float | None] | None,
    t_r_value: Sequence[float | None] | None,
    t_lam: Sequence[float | None] | None,
    t_neurons_sn: Sequence[float | None] | None,
) -> tuple[QcThresholdSpec, ...]:
    requested: dict[str, tuple[float | None, float | None]] = {
        "snr": _normalize_qc_bounds(t_snr, "snr"),
        "r_value": _normalize_qc_bounds(t_r_value, "r_value"),
        "lam": _normalize_qc_bounds(t_lam, "lam"),
        "neurons_sn": _normalize_qc_bounds(t_neurons_sn, "neurons_sn"),
    }
    enabled = {
        key: value
        for key, value in requested.items()
        if value[0] is not None or value[1] is not None
    }
    if not enabled:
        return ()

    stats_path = extract_load_fold / "stats" / "stats.json"
    legacy_stats_path = extract_load_fold / "stats.json"
    if not stats_path.is_file() and legacy_stats_path.is_file():
        stats_path = legacy_stats_path
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"QC thresholds require stats.json, but it was not found: {stats_path}"
        )
    payload = json.loads(stats_path.read_text(encoding="utf-8"))

    specs: list[QcThresholdSpec] = []
    for key, (lower_z, upper_z) in enabled.items():
        raw_stats = payload.get(key)
        if not isinstance(raw_stats, dict):
            raise KeyError(f"Missing QC metric stats for {key}: {stats_path}")
        scale = str(raw_stats["scale"])
        mean = float(raw_stats["mean"])
        std = float(raw_stats["std"])
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Invalid std for QC metric {key} in {stats_path}: {std}")
        specs.append(
            QcThresholdSpec(
                key=key,
                scale=scale,
                mean=mean,
                std=std,
                lower_z=lower_z,
                upper_z=upper_z,
            )
        )
    return tuple(specs)


def _parse_bool_csv(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_profile_metrics(profile_path: Path) -> dict[int, dict[str, float | bool]]:
    rows: dict[int, dict[str, float | bool]] = {}
    with profile_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            component_index = int(row["component_index"])
            metrics: dict[str, float | bool] = {}
            for key in QUALITY_PROFILE_KEYS:
                if key == "r_value_unreliable_joint_only":
                    metrics[key] = _parse_bool_csv(row[key])
                else:
                    metrics[key] = float(row[key])
            rows[component_index] = metrics
    return rows


def _read_patch_qc_keep_mask(
    profile_path: Path,
    n_components: int,
    qc_threshold_specs: Sequence[QcThresholdSpec],
) -> np.ndarray:
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing profile CSV for web QC: {profile_path}")

    if not qc_threshold_specs:
        return np.ones(n_components, dtype=bool)

    keep = np.ones(n_components, dtype=bool)
    seen = 0
    with profile_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            component_index = int(row["component_index"])
            if component_index < 0 or component_index >= n_components:
                raise ValueError(
                    f"component_index out of range in {profile_path}: "
                    f"{component_index} vs n_components={n_components}"
                )
            keep_value = True
            for spec in qc_threshold_specs:
                metric_space_value = _metric_to_qc_space(
                    raw_value=float(row[spec.key]),
                    scale=spec.scale,
                )
                z_value = (metric_space_value - spec.mean) / spec.std
                if spec.lower_z is not None and z_value < spec.lower_z:
                    keep_value = False
                    break
                if spec.upper_z is not None and z_value > spec.upper_z:
                    keep_value = False
                    break
            keep[component_index] = keep_value
            seen += 1
    if seen != n_components:
        raise ValueError(
            f"Profile/model component mismatch for {profile_path}: "
            f"csv_rows={seen}, n_components={n_components}"
        )
    return keep


def _component_core_coordinates_and_weights(spatial, patch: PatchMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_idx = spatial.indices.astype(np.int64, copy=False)
    weights = spatial.data.astype(np.float32, copy=False)
    ys_full = flat_idx % patch.h
    xs_full = flat_idx // patch.h
    core_mask = (
        (ys_full >= patch.core_ly0)
        & (ys_full < patch.core_ly1)
        & (xs_full >= patch.core_lx0)
        & (xs_full < patch.core_lx1)
    )
    ys_core = ys_full[core_mask] - patch.core_ly0
    xs_core = xs_full[core_mask] - patch.core_lx0
    weights_core = weights[core_mask]
    return (
        ys_core.astype(np.int32, copy=False),
        xs_core.astype(np.int32, copy=False),
        weights_core.astype(np.float32, copy=False),
    )


def _component_outline_coordinates(
    spatial,
    patch: PatchMeta,
    support_threshold_rel: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ys_core, xs_core, weights_core = _component_core_coordinates_and_weights(spatial, patch)
    if ys_core.size == 0:
        raise ValueError(f"Component has no support inside patch core: {patch.patch_name}")

    peak_index = int(np.argmax(weights_core))
    peak_y_core = int(ys_core[peak_index])
    peak_x_core = int(xs_core[peak_index])

    core_mask = np.zeros((patch.core_h, patch.core_w), dtype=bool)
    threshold = float(np.max(weights_core)) * float(support_threshold_rel)
    core_mask[ys_core, xs_core] = weights_core >= threshold
    if not core_mask[peak_y_core, peak_x_core]:
        core_mask[peak_y_core, peak_x_core] = True

    labeled, _ = ndi.label(core_mask)
    peak_label = int(labeled[peak_y_core, peak_x_core])
    if peak_label > 0:
        core_mask = labeled == peak_label

    eroded = ndi.binary_erosion(core_mask, structure=np.ones((3, 3), dtype=bool))
    outline = core_mask & ~eroded
    if not np.any(outline):
        outline[peak_y_core, peak_x_core] = True

    outline_y_core, outline_x_core = np.nonzero(outline)
    outline_y = outline_y_core.astype(np.int32, copy=False) + int(patch.y0)
    outline_x = outline_x_core.astype(np.int32, copy=False) + int(patch.x0)
    return (
        outline_y.astype(np.int32, copy=False),
        outline_x.astype(np.int32, copy=False),
        int(patch.y0 + peak_y_core),
        int(patch.x0 + peak_x_core),
    )


def _global_peak_from_sparse_argmax(spatial, patch: PatchMeta) -> tuple[int, int]:
    if spatial.nnz == 0:
        raise ValueError("Cannot derive fallback peak for empty spatial component.")
    data = np.asarray(spatial.data).reshape(-1)
    local_flat_index = int(spatial.indices[int(np.argmax(data))])
    local_y, local_x = np.unravel_index(local_flat_index, (patch.h, patch.w))
    global_y = int(patch.y0 - patch.core_ly0 + local_y)
    global_x = int(patch.x0 - patch.core_lx0 + local_x)
    return global_y, global_x


def _imagej_auto_contrast_to_uint8(
    image: np.ndarray,
    saturated_percent: float = 0.35,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, saturated_percent))
    hi = float(np.percentile(finite, 100.0 - saturated_percent))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _write_background_png(bg_load_path: Path, output_path: Path) -> dict[str, object]:
    image = tifffile.imread(bg_load_path)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D background image, got {image.shape}: {bg_load_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_uint8 = _imagej_auto_contrast_to_uint8(image)
    Image.fromarray(png_uint8, mode="L").save(output_path)
    return {
        "file": str(output_path.relative_to(output_path.parent.parent)).replace("\\", "/"),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "source_path": str(bg_load_path),
        "key": "background",
        "label": bg_load_path.name,
        "contrast_mode": "imagej_auto_0p35pct",
    }


def _infer_dataset_root(extract_load_fold: Path) -> Path:
    dataset_root = extract_load_fold.parent
    if not extract_load_fold.is_dir():
        raise NotADirectoryError(f"Missing extract folder: {extract_load_fold}")
    if not (dataset_root / "Y").is_dir():
        raise NotADirectoryError(
            f"Missing patch movie folder inferred from extract folder: {dataset_root / 'Y'}"
        )
    if not (dataset_root / "crop").is_dir():
        raise NotADirectoryError(
            f"Missing crop folder inferred from extract folder: {dataset_root / 'crop'}"
        )
    return dataset_root


def _prepare_patches(extract_load_fold: Path) -> tuple[Path, list[PatchMeta], int, int]:
    dataset_root = _infer_dataset_root(extract_load_fold)
    y_load_fold = dataset_root / "Y"
    model_dir = get_model_dir(extract_load_fold)
    crop_dir = dataset_root / "crop"
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Missing model folder: {model_dir}")

    patches = _collect_patch_meta(y_load_fold, model_dir)
    crop_params = _load_crop_params(crop_dir)
    core_specs = {
        spec.name: (int(spec.y0), int(spec.y1), int(spec.x0), int(spec.x1))
        for spec in crop_params["patch_specs"]
    }
    full_h = max(v[1] for v in core_specs.values())
    full_w = max(v[3] for v in core_specs.values())

    for patch in patches:
        if patch.patch_name not in core_specs:
            raise KeyError(f"Missing crop spec for patch: {patch.patch_name}")
        y0, y1, x0, x1 = core_specs[patch.patch_name]
        core_h = int(y1 - y0)
        core_w = int(x1 - x0)
        pad_h = int(patch.h - core_h)
        pad_w = int(patch.w - core_w)
        if core_h <= 0 or core_w <= 0:
            raise ValueError(f"Invalid core size for {patch.patch_name}: {(core_h, core_w)}")
        if pad_h < 0 or pad_w < 0 or (pad_h % 2) != 0 or (pad_w % 2) != 0:
            raise ValueError(
                f"Invalid patch/core alignment for {patch.patch_name}: "
                f"patch={(patch.h, patch.w)}, core={(core_h, core_w)}"
            )
        patch.y0, patch.y1, patch.x0, patch.x1 = y0, y1, x0, x1
        patch.core_h, patch.core_w = core_h, core_w
        patch.core_ly0 = pad_h // 2
        patch.core_ly1 = patch.core_ly0 + core_h
        patch.core_lx0 = pad_w // 2
        patch.core_lx1 = patch.core_lx0 + core_w

    return dataset_root, patches, full_h, full_w


def _compute_trace_stats(traces: np.ndarray) -> dict[str, np.ndarray]:
    mean = traces.mean(axis=1, dtype=np.float64).astype(np.float32)
    std = traces.std(axis=1, dtype=np.float64).astype(np.float32)
    p05 = np.percentile(traces, 5.0, axis=1).astype(np.float32)
    p95 = np.percentile(traces, 95.0, axis=1).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "p05": p05,
        "p95": p95,
    }


def _json_float_or_none(value: float | np.floating) -> float | None:
    scalar = float(value)
    if np.isfinite(scalar):
        return scalar
    return None


def _load_web_qc_thresholds() -> dict[str, list[float | None] | None]:
    if not SAVE_SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Missing save schema for QC thresholds: {SAVE_SCHEMA_PATH}")
    payload = yaml.safe_load(SAVE_SCHEMA_PATH.read_text(encoding="utf-8"))
    save_cfg = payload.get("save") if isinstance(payload, dict) else None
    if not isinstance(save_cfg, dict):
        raise ValueError(f"Invalid save schema structure: {SAVE_SCHEMA_PATH}")
    return {
        "t_snr": save_cfg.get("t_snr"),
        "t_r_value": save_cfg.get("t_r_value"),
        "t_lam": save_cfg.get("t_lam"),
        "t_neurons_sn": save_cfg.get("t_neurons_sn"),
    }


def _suppress_noisy_loggers() -> None:
    for logger_name in ("caiman", "caiman.source_extraction", "web.build_cache"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False


def build_cache(
    *,
    bg_load_path: Path,
    extract_load_fold: Path,
    app_fold: Path,
) -> None:
    _suppress_noisy_loggers()
    extract_load_fold = extract_load_fold.resolve()
    bg_load_path = bg_load_path.resolve()
    app_fold = app_fold.resolve()
    app_fold.mkdir(parents=True, exist_ok=True)
    (app_fold / BACKGROUND_DIRNAME).mkdir(parents=True, exist_ok=True)

    dataset_root, patches, full_h, full_w = _prepare_patches(extract_load_fold)
    if not bg_load_path.is_file():
        raise FileNotFoundError(f"Missing background image: {bg_load_path}")

    qc_thresholds = _load_web_qc_thresholds()
    qc_threshold_specs = _load_qc_threshold_specs(
        extract_load_fold=extract_load_fold,
        t_snr=qc_thresholds["t_snr"],
        t_r_value=qc_thresholds["t_r_value"],
        t_lam=qc_thresholds["t_lam"],
        t_neurons_sn=qc_thresholds["t_neurons_sn"],
    )

    background_output = app_fold / BACKGROUND_DIRNAME / "background.png"
    background_specs = [_write_background_png(bg_load_path, background_output)]

    all_trace_rows: list[np.ndarray] = []
    xs: list[int] = []
    ys: list[int] = []
    ids: list[int] = []
    patch_names: list[str] = []
    component_indices: list[int] = []
    metric_columns: dict[str, list[float | bool]] = {key: [] for key in QUALITY_PROFILE_KEYS}

    neuron_id = 0
    trace_length: int | None = None
    frame_rate_hz: float | None = None
    anchor_counts = {"outline": 0, "fallback_argmax": 0}

    model_dir = get_model_dir(extract_load_fold)
    profile_dir = get_profile_dir(extract_load_fold)

    for patch in tqdm(patches, desc="app", unit="patch", dynamic_ncols=True):
        model_path = model_dir / f"{patch.patch_name}.tif.hdf5"
        profile_path = profile_dir / f"{patch.patch_name}.tif.csv"
        if not model_path.is_file() or not profile_path.is_file():
            continue

        metrics_by_component = _read_profile_metrics(profile_path)
        with silence_stdio():
            cnm_model = load_CNMF(str(model_path), n_processes=1, dview=None)

        A = cnm_model.estimates.A.tocsc()
        C = np.asarray(cnm_model.estimates.C, dtype=np.float32)
        if trace_length is None:
            trace_length = int(C.shape[1])
        elif trace_length != int(C.shape[1]):
            raise ValueError(f"Unexpected trace length mismatch at {patch.patch_name}")

        patch_frame_rate_hz = float(cnm_model.params.get("data", "fr"))
        if frame_rate_hz is None:
            frame_rate_hz = patch_frame_rate_hz
        elif abs(frame_rate_hz - patch_frame_rate_hz) > 1e-6:
            raise ValueError("Frame-rate mismatch across patches.")

        if A.shape[1] != len(metrics_by_component):
            raise ValueError(
                f"Profile/model mismatch for {patch.patch_name}: A={A.shape[1]}, csv={len(metrics_by_component)}"
            )

        keep_mask = _read_patch_qc_keep_mask(
            profile_path=profile_path,
            n_components=A.shape[1],
            qc_threshold_specs=qc_threshold_specs,
        )

        for component_index in range(A.shape[1]):
            if not bool(keep_mask[component_index]):
                continue
            spatial = A.getcol(component_index)
            try:
                _, _, peak_y, peak_x = _component_outline_coordinates(spatial, patch)
                anchor_counts["outline"] += 1
            except ValueError:
                peak_y, peak_x = _global_peak_from_sparse_argmax(spatial, patch)
                anchor_counts["fallback_argmax"] += 1

            metrics = metrics_by_component[component_index]
            ids.append(neuron_id)
            xs.append(int(peak_x))
            ys.append(int(peak_y))
            patch_names.append(patch.patch_name)
            component_indices.append(int(component_index))
            for key in QUALITY_PROFILE_KEYS:
                metric_columns[key].append(metrics[key])
            all_trace_rows.append(np.asarray(C[component_index], dtype=np.float32))
            neuron_id += 1

    if not all_trace_rows:
        raise RuntimeError("No renderable neurons found for web cache.")
    if trace_length is None or frame_rate_hz is None:
        raise RuntimeError("Failed to infer trace metadata.")

    traces = np.stack(all_trace_rows, axis=0).astype(np.float32, copy=False)
    trace_stats = _compute_trace_stats(traces)
    trace_file = app_fold / TRACE_FILE_NAME
    traces.tofile(trace_file)

    points_payload = {
        "id": ids,
        "x": xs,
        "y": ys,
        "patch_name": patch_names,
        "component_index": component_indices,
        "metrics": {
            key: [bool(v) if key == "r_value_unreliable_joint_only" else _json_float_or_none(v) for v in values]
            for key, values in metric_columns.items()
        },
        "trace_stats": {
            TRACE_SOURCE_KEY: {
                stat_key: [_json_float_or_none(v) for v in values.astype(np.float32)]
                for stat_key, values in trace_stats.items()
            }
        },
    }
    points_file = app_fold / POINTS_FILE_NAME
    points_file.write_text(json.dumps(points_payload, separators=(",", ":")), encoding="utf-8")

    metadata = {
        "dataset_root": str(dataset_root),
        "extract_load_fold": str(extract_load_fold),
        "bg_load_path": str(bg_load_path),
        "app_fold": str(app_fold),
        "full_height": int(full_h),
        "full_width": int(full_w),
        "trace_length": int(trace_length),
        "frame_rate_hz": float(frame_rate_hz),
        "neuron_count": int(len(ids)),
        "trace_sources": {
            TRACE_SOURCE_KEY: {
                "file": TRACE_FILE_NAME,
                "label": "C",
                "description": "CNMF fitted temporal trace",
                "dtype": "float32",
            }
        },
        "points_file": POINTS_FILE_NAME,
        "backgrounds": background_specs,
        "default_background_key": "background",
        "anchor_counts": anchor_counts,
        "qc_filtering": {
            "enabled": True,
            "source": str(SAVE_SCHEMA_PATH),
            "thresholds": qc_thresholds,
        },
    }
    metadata_file = app_fold / METADATA_FILE_NAME
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cache files for the standalone neuron ROI web app.")
    parser.add_argument("--bg_load_path", type=Path, required=True, help="Path to the background TIFF.")
    parser.add_argument("--extract_load_fold", type=Path, required=True, help="Path to the extract folder.")
    parser.add_argument("--app_fold", type=Path, required=True, help="Path to the app cache folder.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    build_cache(
        bg_load_path=args.bg_load_path,
        extract_load_fold=args.extract_load_fold,
        app_fold=args.app_fold,
    )


if __name__ == "__main__":
    main()
