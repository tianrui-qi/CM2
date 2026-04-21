import csv
import json
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
import logging
import os
import re
import shutil

import numpy as np
from scipy import fft as scipy_fft
import tifffile
import tqdm
import zarr

from .crop import (
    load_crop_params,
)


PATCH_NAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)$")
MODEL_STATIC_OUTPUT_ORDER = ("Bc", "ACsum", "ACYrAsum")
MODEL_VIDEO_OUTPUT_ORDER = ("AC", "ACYrA", "Bf", "B")
MODEL_QC_STATIC_OUTPUT_ORDER = ("ACsum-qc", "ACYrAsum-qc")
MODEL_QC_VIDEO_OUTPUT_ORDER = ("AC-qc", "ACYrA-qc")
MODEL_2D_OUTPUT_ORDER = MODEL_STATIC_OUTPUT_ORDER + MODEL_QC_STATIC_OUTPUT_ORDER
MODEL_3D_OUTPUT_ORDER = MODEL_VIDEO_OUTPUT_ORDER + MODEL_QC_VIDEO_OUTPUT_ORDER
MODEL_OUTPUT_ORDER = (
    MODEL_2D_OUTPUT_ORDER
    + MODEL_3D_OUTPUT_ORDER
)
LEGACY_TEMP_OUTPUT_DIRS = ("A", "C", "YrA", "W", "Bc") + MODEL_OUTPUT_ORDER
MODEL_BAR_LABEL_WIDTH = 5
IMAGEJ_BANDPASS_FILTER_LARGE_DIAMETER_PX = 5.0
IMAGEJ_BANDPASS_FILTER_SMALL_DIAMETER_PX = 1.0
POINTMAP_DIRNAME = "figure-PointMap"
POINTMAP_LABEL_DIRNAME = "figure-PointMap-label"
THRESHOLDSTACK_DIRNAME = "figure-ThresholdStack"
OVERLAP_POINTMAP_DIRNAME = "overlap-PointMap"
OVERLAP_POINTMAP_LABEL_DIRNAME = "overlap-PointMap-label"
OVERLAP_THRESHOLDSTACK_DIRNAME = "overlap-ThresholdStack"


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
class QcThresholdSpec:
    key: str
    scale: str
    mean: float
    std: float
    lower_z: float | None
    upper_z: float | None


def _parse_patch_name(stem: str) -> tuple[int, int, int, int]:
    m = PATCH_NAME_RE.match(stem)
    if m is None:
        raise ValueError(
            f"Patch name must match '<qrow>-<qcol>-<prow>-<pcol>', got: {stem}"
        )
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def _model_file_to_patch_name(path: Path) -> str:
    name = path.name
    if name.endswith(".tif.hdf5"):
        return name[: -len(".tif.hdf5")]
    if name.endswith(".hdf5"):
        return path.stem
    raise ValueError(f"Unsupported model filename: {name}")


def _resolve_patch_folders(
    y_load_path: Path,
    extract_fold_cfg: str | None,
) -> tuple[Path, Path]:
    y_load_fold = y_load_path.with_suffix("")
    if not y_load_fold.is_dir():
        fallback = y_load_path.parent / "Y"
        if fallback.is_dir():
            y_load_fold = fallback
        else:
            raise FileNotFoundError(
                f"Patch movie folder not found. Tried: {y_load_fold} and {fallback}"
            )

    extract_fold = (
        y_load_path.parent / "extract"
        if extract_fold_cfg is None
        else Path(extract_fold_cfg)
    )
    if not extract_fold.is_dir():
        raise NotADirectoryError(
            f"extract_fold must be an existing directory: {extract_fold}"
        )
    model_dir = extract_fold / "patch-model"
    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"extract_fold must contain a patch-model directory: {model_dir}"
        )
    return y_load_fold, model_dir


def _collect_patch_meta(
    y_load_fold: Path,
    extract_fold: Path,
) -> list[PatchMeta]:
    patches: list[PatchMeta] = []
    model_files = sorted(extract_fold.glob("*.hdf5"), key=lambda p: p.name)
    if not model_files:
        raise FileNotFoundError(f"No .hdf5 model files found in: {extract_fold}")

    for model_path in model_files:
        patch_name = _model_file_to_patch_name(model_path)
        qrow, qcol, prow, pcol = _parse_patch_name(patch_name)
        movie_path = y_load_fold / f"{patch_name}.tif"
        if not movie_path.exists():
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


def _resolve_extract_root(model_dir_or_root: Path) -> Path:
    if model_dir_or_root.name == "patch-model":
        return model_dir_or_root.parent
    return model_dir_or_root


def _metric_to_qc_space(
    raw_value: float,
    scale: str,
) -> float:
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
    extract_root: Path,
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

    stats_path = extract_root / "stats" / "stats.json"
    if not stats_path.is_file():
        legacy_stats_path = extract_root / "stats.json"
        if legacy_stats_path.is_file():
            stats_path = legacy_stats_path
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"QC thresholds require extract stats.json, but it was not found: {stats_path}"
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
            raise ValueError(
                f"Invalid std for QC metric {key} in {stats_path}: {std}"
            )
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


def _read_patch_qc_keep_mask(
    profile_path: Path,
    n_components: int,
    qc_threshold_specs: Sequence[QcThresholdSpec],
) -> np.ndarray:
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing profile CSV for QC save: {profile_path}")

    if not qc_threshold_specs:
        return np.ones(n_components, dtype=bool)

    keep = np.ones(n_components, dtype=bool)
    seen = 0
    with profile_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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


def _count_profile_qc_components(
    profile_path: Path,
    qc_threshold_specs: Sequence[QcThresholdSpec],
) -> tuple[int, int]:
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing profile CSV for QC count: {profile_path}")

    total = 0
    kept = 0
    with profile_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
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
            if keep_value:
                kept += 1
    return total, kept


def _summarize_qc_component_counts(
    extract_root: Path,
    t_snr: Sequence[float | None] | None,
    t_r_value: Sequence[float | None] | None,
    t_lam: Sequence[float | None] | None,
    t_neurons_sn: Sequence[float | None] | None,
) -> tuple[int, int] | None:
    profile_dir = extract_root / "patch-profile"
    if not profile_dir.is_dir():
        return None

    qc_threshold_specs = _load_qc_threshold_specs(
        extract_root=extract_root,
        t_snr=t_snr,
        t_r_value=t_r_value,
        t_lam=t_lam,
        t_neurons_sn=t_neurons_sn,
    )

    profile_paths = sorted(
        path
        for path in profile_dir.glob("*.tif.csv")
        if not path.name.startswith("._")
    )
    total_before = 0
    total_after = 0
    for profile_path in profile_paths:
        total, kept = _count_profile_qc_components(
            profile_path=profile_path,
            qc_threshold_specs=qc_threshold_specs,
        )
        total_before += total
        total_after += kept
    return total_before, total_after


def _compute_patch_layout(
    patches: list[PatchMeta],
) -> tuple[int, int, dict[str, tuple[int, int, int, int]]]:
    groups: dict[tuple[int, int], list[PatchMeta]] = {}
    for p in patches:
        groups.setdefault((p.qrow, p.qcol), []).append(p)

    local_pos: dict[str, tuple[int, int]] = {}
    quad_wh: dict[tuple[int, int], tuple[int, int]] = {}

    for qkey, items in groups.items():
        rows = sorted({x.prow for x in items})
        cols = sorted({x.pcol for x in items})
        row_index = {r: i for i, r in enumerate(rows)}
        col_index = {c: i for i, c in enumerate(cols)}

        row_heights: list[int] = []
        for r in rows:
            row_items = [x for x in items if x.prow == r]
            got_cols = sorted({x.pcol for x in row_items})
            if got_cols != cols:
                raise ValueError(
                    f"Quadrant {qkey} row {r} missing columns, got={got_cols}, expected={cols}"
                )
            hs = sorted({x.h for x in row_items})
            if len(hs) != 1:
                raise ValueError(
                    f"Quadrant {qkey} row {r} has inconsistent heights: {hs}"
                )
            row_heights.append(hs[0])

        col_widths: list[int] = []
        for c in cols:
            col_items = [x for x in items if x.pcol == c]
            got_rows = sorted({x.prow for x in col_items})
            if got_rows != rows:
                raise ValueError(
                    f"Quadrant {qkey} col {c} missing rows, got={got_rows}, expected={rows}"
                )
            ws = sorted({x.w for x in col_items})
            if len(ws) != 1:
                raise ValueError(
                    f"Quadrant {qkey} col {c} has inconsistent widths: {ws}"
                )
            col_widths.append(ws[0])

        y_off = [0]
        for h in row_heights[:-1]:
            y_off.append(y_off[-1] + h)
        x_off = [0]
        for w in col_widths[:-1]:
            x_off.append(x_off[-1] + w)

        for it in items:
            local_pos[it.patch_name] = (
                y_off[row_index[it.prow]],
                x_off[col_index[it.pcol]],
            )

        quad_wh[qkey] = (sum(row_heights), sum(col_widths))

    qrows = sorted({p.qrow for p in patches})
    qcols = sorted({p.qcol for p in patches})
    row_heights_global = {
        qr: max(quad_wh[(qr, qc)][0] for qc in qcols if (qr, qc) in quad_wh)
        for qr in qrows
    }
    col_widths_global = {
        qc: max(quad_wh[(qr, qc)][1] for qr in qrows if (qr, qc) in quad_wh)
        for qc in qcols
    }

    y_base: dict[int, int] = {}
    cur = 0
    for qr in qrows:
        y_base[qr] = cur
        cur += row_heights_global[qr]
    full_h = cur

    x_base: dict[int, int] = {}
    cur = 0
    for qc in qcols:
        x_base[qc] = cur
        cur += col_widths_global[qc]
    full_w = cur

    placements: dict[str, tuple[int, int, int, int]] = {}
    for p in patches:
        ly, lx = local_pos[p.patch_name]
        y0 = y_base[p.qrow] + ly
        x0 = x_base[p.qcol] + lx
        placements[p.patch_name] = (y0, y0 + p.h, x0, x0 + p.w)

    return full_h, full_w, placements


def _get_available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        avail = int(psutil.virtual_memory().available)
        if avail > 0:
            return avail
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            if ok:
                avail = int(stat.ullAvailPhys)
                if avail > 0:
                    return avail
        except Exception:
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        avail = page_size * avail_pages
        if avail > 0:
            return avail
    except Exception:
        return None
    return None


def _create_output_memmap(
    path: Path,
    shape: tuple[int, ...],
    zero_init: bool,
) -> np.memmap:
    if path.exists():
        path.unlink()
    array = tifffile.memmap(
        path,
        shape=shape,
        dtype=np.float32,
        photometric="minisblack",
        bigtiff=(int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize) >= (2**31),
    )
    if zero_init:
        array[...] = 0.0
    return array


def _grid_has_holes(
    patches: list[PatchMeta],
    full_h: int,
    full_w: int,
) -> bool:
    covered = np.zeros((full_h, full_w), dtype=bool)
    for p in patches:
        covered[p.y0 : p.y1, p.x0 : p.x1] = True
    return not bool(np.all(covered))


def _choose_model_chunk(
    p: PatchMeta,
    global_t: int,
    need_ac: bool,
    need_acyra: bool,
    need_background: bool,
) -> int:
    if global_t <= 0:
        return 1

    pixel_count = max(1, p.h * p.w)
    bytes_per_frame = pixel_count * np.dtype(np.float32).itemsize

    available_memory = _get_available_memory_bytes()
    if available_memory is None:
        scratch_budget = 512 * 1024 * 1024
        available_memory = scratch_budget

    reserve_bytes = max(1024**3, int(available_memory * 0.12))
    working_budget = available_memory - reserve_bytes
    if working_budget <= 0:
        return 1

    live_array_count = 0.0
    if need_ac or need_background:
        live_array_count += 1.0  # AC matrix
    if need_acyra:
        live_array_count += 1.0  # ACYrA matrix
    if need_background:
        live_array_count += 3.0  # movie block, residual, Bf
    if live_array_count <= 0:
        return global_t

    # Keep a cushion for sparse matmul internals, reshapes/views, and Python overhead
    live_array_count *= 1.25
    max_frames = int(working_budget // max(1.0, bytes_per_frame * live_array_count))
    if max_frames <= 0:
        return 1
    return max(1, min(global_t, max_frames))


def _choose_stats_chunk(
    t: int,
    h: int,
    w: int,
    save_std: bool,
) -> tuple[int, bool]:
    if t <= 0:
        return 1, True

    pixel_count = max(1, h * w)
    float32_bytes = np.dtype(np.float32).itemsize
    float64_bytes = np.dtype(np.float64).itemsize
    accumulator_bytes = pixel_count * float64_bytes * (1 + int(save_std))
    mean_work_bytes = pixel_count * float64_bytes
    per_frame_scratch_bytes = pixel_count * float32_bytes

    available_memory = _get_available_memory_bytes()
    if available_memory is None:
        scratch_budget = 256 * 1024 * 1024
        max_chunk = max(1, scratch_budget // max(1, per_frame_scratch_bytes))
        return max(1, min(t, max_chunk)), True

    reserve_bytes = max(1024**3, int(available_memory * 0.12))
    working_budget = available_memory - reserve_bytes - accumulator_bytes - mean_work_bytes
    if working_budget < per_frame_scratch_bytes:
        fallback_chunk = max(1, min(t, 8))
        return fallback_chunk, False

    max_chunk = max(1, int(working_budget // max(1, per_frame_scratch_bytes)))
    return max(1, min(t, max_chunk)), True


def _read_tiff(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.round(arr.astype(np.float32) / 257.0).astype(np.uint8)
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _imagej_auto_display_range_once(image: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0

    hist_min = float(np.min(finite))
    hist_max = float(np.max(finite))
    if not np.isfinite(hist_min) or not np.isfinite(hist_max) or hist_max <= hist_min:
        return hist_min, hist_max

    histogram, _ = np.histogram(finite, bins=256, range=(hist_min, hist_max))
    pixel_count = int(finite.size)
    limit = pixel_count // 10
    threshold = pixel_count // 5000

    i = -1
    found = False
    while (not found) and (i < 255):
        i += 1
        count = int(histogram[i])
        if count > limit:
            count = 0
        found = count > threshold
    hmin = i

    i = 256
    found = False
    while (not found) and (i > 0):
        i -= 1
        count = int(histogram[i])
        if count > limit:
            count = 0
        found = count > threshold
    hmax = i

    if hmax < hmin:
        return hist_min, hist_max

    bin_size = (hist_max - hist_min) / 256.0
    display_min = hist_min + hmin * bin_size
    display_max = hist_min + hmax * bin_size
    if display_max <= display_min:
        return hist_min, hist_max
    return float(display_min), float(display_max)


def _normalize_to_uint8_imagej_auto(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    display_min, display_max = _imagej_auto_display_range_once(arr)
    if not np.isfinite(display_min) or not np.isfinite(display_max) or display_max <= display_min:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - display_min) / (display_max - display_min), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _background_to_rgb_uint8(background: np.ndarray) -> np.ndarray:
    if background.ndim == 2:
        gray = _normalize_to_uint8_imagej_auto(background)
        return np.repeat(gray[..., None], 3, axis=-1)
    if background.ndim == 3 and background.shape[-1] in (3, 4):
        rgb = background[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        if rgb.dtype == np.uint16:
            return np.round(rgb.astype(np.float32) / 257.0).astype(np.uint8)
        channels = [
            _normalize_to_uint8_imagej_auto(rgb[..., channel_idx])
            for channel_idx in range(3)
        ]
        return np.stack(channels, axis=-1)
    raise ValueError(
        f"Background must be 2D grayscale or YXS RGB/RGBA, got shape={background.shape}"
    )


def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    if src_h == target_h and src_w == target_w:
        return image
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * src_h) // target_h, src_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * src_w) // target_w, src_w - 1)
    if image.ndim == 2:
        return image[y_idx][:, x_idx]
    if image.ndim == 3:
        return image[y_idx][:, x_idx, :]
    raise ValueError(f"Unsupported resize image ndim={image.ndim}")


def _overlay_rgb(background_rgb: np.ndarray, foreground_rgb_uint8: np.ndarray) -> np.ndarray:
    out = background_rgb.copy()
    mask = np.any(foreground_rgb_uint8 != 0, axis=-1)
    out[mask] = foreground_rgb_uint8[mask]
    return out


def _overlay_stack(background_rgb: np.ndarray, foreground_stack_uint8: np.ndarray) -> np.ndarray:
    out = np.repeat(background_rgb[None, ...], foreground_stack_uint8.shape[0], axis=0)
    mask = np.any(foreground_stack_uint8 != 0, axis=-1)
    out[mask] = foreground_stack_uint8[mask]
    return out


def _list_metric_tifs(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.glob("*.tif") if path.is_file() and not path.name.startswith("._")]
    )


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _overlay_folder(
    source_dir: Path,
    output_dir: Path,
    background_rgb_base: np.ndarray,
    preserve_labels: bool,
    progress_bar: tqdm.tqdm | None = None,
) -> None:
    _prepare_output_dir(output_dir)
    for source_path in _list_metric_tifs(source_dir):
        with tifffile.TiffFile(source_path) as tif:
            foreground = tif.asarray()
            labels = None
            if preserve_labels:
                imagej_meta = tif.imagej_metadata or {}
                labels = imagej_meta.get("Labels")

        if foreground.ndim == 3 and foreground.shape[-1] == 3:
            target_h, target_w = int(foreground.shape[0]), int(foreground.shape[1])
            background_rgb = _resize_nearest(background_rgb_base, target_h, target_w)
            out = _overlay_rgb(background_rgb, foreground)
            tifffile.imwrite(output_dir / source_path.name, out, photometric="rgb")
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        if foreground.ndim == 4 and foreground.shape[-1] == 3:
            target_h, target_w = int(foreground.shape[1]), int(foreground.shape[2])
            background_rgb = _resize_nearest(background_rgb_base, target_h, target_w)
            out = _overlay_stack(background_rgb, foreground)
            if labels is not None and len(labels) == int(out.shape[0]):
                tifffile.imwrite(
                    output_dir / source_path.name,
                    out,
                    imagej=True,
                    photometric="rgb",
                    metadata={"Labels": labels},
                )
            else:
                tifffile.imwrite(
                    output_dir / source_path.name,
                    out,
                    imagej=True,
                    photometric="rgb",
                )
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        raise ValueError(f"Unsupported foreground shape for {source_path}: {foreground.shape}")


def save_overlap_outputs(
    extract_root: Path,
    background_path: Path,
) -> None:
    if not background_path.is_file():
        raise FileNotFoundError(f"Background not found: {background_path}")
    if not extract_root.is_dir():
        raise NotADirectoryError(f"Extract folder not found: {extract_root}")

    background = _read_tiff(background_path)
    background_rgb_base = _background_to_rgb_uint8(background)

    pointmap_dir = extract_root / POINTMAP_DIRNAME
    pointmap_label_dir = extract_root / POINTMAP_LABEL_DIRNAME
    thresholdstack_dir = extract_root / THRESHOLDSTACK_DIRNAME
    for required_dir in (pointmap_dir, pointmap_label_dir, thresholdstack_dir):
        if not required_dir.is_dir():
            raise NotADirectoryError(f"Missing figure folder: {required_dir}")

    total_figures = (
        len(_list_metric_tifs(pointmap_dir))
        + len(_list_metric_tifs(pointmap_label_dir))
        + len(_list_metric_tifs(thresholdstack_dir))
    )
    with tqdm.tqdm(
        total=total_figures,
        desc="save(overlap)",
        dynamic_ncols=True,
    ) as progress_bar:
        _overlay_folder(
            pointmap_dir,
            extract_root / OVERLAP_POINTMAP_DIRNAME,
            background_rgb_base,
            preserve_labels=False,
            progress_bar=progress_bar,
        )
        _overlay_folder(
            pointmap_label_dir,
            extract_root / OVERLAP_POINTMAP_LABEL_DIRNAME,
            background_rgb_base,
            preserve_labels=False,
            progress_bar=progress_bar,
        )
        _overlay_folder(
            thresholdstack_dir,
            extract_root / OVERLAP_THRESHOLDSTACK_DIRNAME,
            background_rgb_base,
            preserve_labels=True,
            progress_bar=progress_bar,
        )


def _next_power_of_two_at_least(value: float) -> int:
    n = 2
    while n < value:
        n *= 2
    return n


def _reflect_indices(length: int, out_length: int, offset: int) -> np.ndarray:
    coords = np.arange(out_length, dtype=np.int64) - int(offset)
    period = 2 * int(length)
    mirrored = np.mod(coords, period)
    return np.where(mirrored < length, mirrored, period - 1 - mirrored).astype(np.int64)


def _tile_mirror_imagej(image_yx: np.ndarray, padded_size: int) -> tuple[np.ndarray, int, int]:
    h, w = image_yx.shape
    x0 = int(round((padded_size - w) / 2.0))
    y0 = int(round((padded_size - h) / 2.0))
    xi = _reflect_indices(w, padded_size, x0)
    yi = _reflect_indices(h, padded_size, y0)
    return image_yx[yi][:, xi], x0, y0


def _build_imagej_bandpass_filter(size: int) -> np.ndarray:
    rows = np.minimum(
        np.arange(size, dtype=np.float32),
        size - np.arange(size, dtype=np.float32),
    )
    cols = np.arange(size // 2 + 1, dtype=np.float32)
    scale_large = (2.0 * IMAGEJ_BANDPASS_FILTER_LARGE_DIAMETER_PX / float(size)) ** 2
    scale_small = (2.0 * IMAGEJ_BANDPASS_FILTER_SMALL_DIAMETER_PX / float(size)) ** 2
    radius_sq = rows[:, None] * rows[:, None] + cols[None, :] * cols[None, :]
    bandpass = (1.0 - np.exp(-radius_sq * scale_large)) * np.exp(-radius_sq * scale_small)
    # Match ImageJ FFTFilter.java: DC is preserved.
    bandpass[0, 0] = 1.0
    return bandpass.astype(np.float32, copy=False)


def save_imagej_bandpass_tif(
    input_path: Path,
    output_path: Path,
    progress_label: str | None = None,
) -> None:
    progress_bar = (
        tqdm.tqdm(
            total=1,
            desc=f"save({progress_label})",
            unit="image",
            dynamic_ncols=True,
        )
        if progress_label is not None
        else None
    )
    try:
        image_yx = tifffile.imread(input_path).astype(np.float32, copy=False)
        if image_yx.ndim != 2:
            raise ValueError(f"Expected a single 2D TIFF for bandpass save, got shape={image_yx.shape}: {input_path}")

        h, w = image_yx.shape
        padded_size = _next_power_of_two_at_least(1.5 * max(h, w))
        padded, x0, y0 = _tile_mirror_imagej(image_yx, padded_size)
        bandpass = _build_imagej_bandpass_filter(padded_size)

        spectrum = scipy_fft.rfft2(padded)
        spectrum *= bandpass.astype(spectrum.real.dtype, copy=False)
        filtered = scipy_fft.irfft2(spectrum, s=padded.shape).astype(np.float32, copy=False)
        filtered = filtered[y0 : y0 + h, x0 : x0 + w]
        tifffile.imwrite(output_path, filtered, photometric="minisblack")
        if progress_bar is not None:
            progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()


def _process_single_patch_into_outputs(
    p: PatchMeta,
    global_t: int,
    model_chunk: int,
    static_outputs: dict[str, np.memmap],
    video_outputs: dict[str, np.memmap],
    profile_dir: Path | None,
    qc_threshold_specs: Sequence[QcThresholdSpec],
) -> None:
    from caiman.source_extraction.cnmf.cnmf import load_CNMF

    logging.getLogger("caiman").setLevel(logging.ERROR)
    cnm = load_CNMF(str(p.model_path), n_processes=1, dview=None)
    dims = tuple(int(x) for x in list(cnm.dims)[:2])
    if dims != (p.h, p.w):
        raise ValueError(
            f"Movie/model dims mismatch for {p.patch_name}: movie={(p.h, p.w)}, model={dims}"
        )

    need_bc = ("Bc" in static_outputs) or ("Bf" in video_outputs) or ("B" in video_outputs)
    need_acyra = (
        ("ACYrAsum" in static_outputs)
        or ("ACYrA" in video_outputs)
        or ("ACYrAsum-qc" in static_outputs)
        or ("ACYrA-qc" in video_outputs)
    )
    need_background = ("Bf" in video_outputs) or ("B" in video_outputs)
    need_video = bool(video_outputs)
    need_qc = any(name.endswith("-qc") for name in static_outputs) or any(
        name.endswith("-qc") for name in video_outputs
    )

    A = cnm.estimates.A.tocsr()
    C = np.asarray(cnm.estimates.C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"Expected C as 2D matrix, got shape={C.shape}")
    if A.shape[0] != p.h * p.w or A.shape[1] != C.shape[0]:
        raise ValueError(
            f"A/C shape mismatch for {p.patch_name}: A={A.shape}, C={C.shape}"
        )
    if C.shape[1] < global_t:
        raise ValueError(
            f"C shorter than required global_t for {p.patch_name}: {C.shape[1]} < {global_t}"
        )

    keep_mask: np.ndarray | None = None
    A_qc = None
    C_qc: np.ndarray | None = None
    if need_qc:
        if profile_dir is None:
            raise ValueError(f"QC outputs requested but no profile_dir resolved for {p.patch_name}")
        profile_path = profile_dir / f"{p.patch_name}.tif.csv"
        keep_mask = _read_patch_qc_keep_mask(
            profile_path=profile_path,
            n_components=C.shape[0],
            qc_threshold_specs=qc_threshold_specs,
        )
        if np.any(keep_mask):
            A_qc = A[:, keep_mask]
            C_qc = C[keep_mask, :]
        else:
            C_qc = np.zeros((0, C.shape[1]), dtype=np.float32)

    YrA: np.ndarray | None = None
    YrA_qc: np.ndarray | None = None
    if need_acyra:
        YrA_raw = getattr(cnm.estimates, "YrA", None)
        YrA = (
            np.asarray(YrA_raw, dtype=np.float32)
            if YrA_raw is not None
            else np.zeros_like(C, dtype=np.float32)
        )
        if YrA.shape != C.shape:
            raise ValueError(
                f"YrA/C shape mismatch for {p.patch_name}: YrA={YrA.shape}, C={C.shape}"
            )
        if YrA.shape[1] < global_t:
            raise ValueError(
                f"YrA shorter than required global_t for {p.patch_name}: {YrA.shape[1]} < {global_t}"
            )
        if need_qc:
            if keep_mask is None:
                raise RuntimeError(f"Missing keep_mask while preparing QC YrA for {p.patch_name}")
            if np.any(keep_mask):
                YrA_qc = YrA[keep_mask, :]
            else:
                YrA_qc = np.zeros((0, YrA.shape[1]), dtype=np.float32)

    bc_patch: np.ndarray | None = None
    bc_flat: np.ndarray | None = None
    bc_core: np.ndarray | None = None
    if need_bc:
        b0_raw = getattr(cnm.estimates, "b0", None)
        if b0_raw is None:
            raise ValueError(f"Model has no b0 for {p.patch_name}: {p.model_path}")
        b0 = np.asarray(b0_raw, dtype=np.float32).reshape(-1)
        if b0.size != p.h * p.w:
            raise ValueError(
                f"b0 size mismatch for {p.patch_name}: {b0.size} vs {p.h * p.w}"
            )
        bc_patch = b0.reshape((p.h, p.w), order="F").astype(np.float32, copy=False)
        bc_flat = bc_patch.reshape(-1, order="F")
        bc_core = _crop_core_2d(p, bc_patch)
        if "Bc" in static_outputs:
            static_outputs["Bc"][p.y0 : p.y1, p.x0 : p.x1] = bc_core

    if "ACsum" in static_outputs:
        csum = C[:, :global_t].sum(axis=1, dtype=np.float32)
        acsum_patch = np.asarray(A @ csum, dtype=np.float32).reshape((p.h, p.w), order="F")
        static_outputs["ACsum"][p.y0 : p.y1, p.x0 : p.x1] = _crop_core_2d(p, acsum_patch)

    if "ACsum-qc" in static_outputs:
        if A_qc is None or C_qc is None:
            acsum_qc_patch = np.zeros((p.h, p.w), dtype=np.float32)
        else:
            csum_qc = C_qc[:, :global_t].sum(axis=1, dtype=np.float32)
            acsum_qc_patch = np.asarray(A_qc @ csum_qc, dtype=np.float32).reshape(
                (p.h, p.w),
                order="F",
            )
        static_outputs["ACsum-qc"][p.y0 : p.y1, p.x0 : p.x1] = _crop_core_2d(
            p,
            acsum_qc_patch,
        )

    if "ACYrAsum" in static_outputs:
        if YrA is None:
            raise RuntimeError(f"Missing YrA while writing ACYrAsum for {p.patch_name}")
        acyra_sum = (C[:, :global_t] + YrA[:, :global_t]).sum(axis=1, dtype=np.float32)
        acyrasum_patch = np.asarray(A @ acyra_sum, dtype=np.float32).reshape((p.h, p.w), order="F")
        static_outputs["ACYrAsum"][p.y0 : p.y1, p.x0 : p.x1] = _crop_core_2d(p, acyrasum_patch)

    if "ACYrAsum-qc" in static_outputs:
        if A_qc is None or C_qc is None or YrA_qc is None:
            acyrasum_qc_patch = np.zeros((p.h, p.w), dtype=np.float32)
        else:
            acyra_sum_qc = (C_qc[:, :global_t] + YrA_qc[:, :global_t]).sum(
                axis=1,
                dtype=np.float32,
            )
            acyrasum_qc_patch = np.asarray(A_qc @ acyra_sum_qc, dtype=np.float32).reshape(
                (p.h, p.w),
                order="F",
            )
        static_outputs["ACYrAsum-qc"][p.y0 : p.y1, p.x0 : p.x1] = _crop_core_2d(
            p,
            acyrasum_qc_patch,
        )

    if not need_video:
        return

    W = None
    if need_background:
        W = getattr(cnm.estimates, "W", None)
        if W is None:
            raise ValueError(f"Model has no W for {p.patch_name}: {p.model_path}")
        W = W.tocsr()

    with tifffile.TiffFile(p.movie_path) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim != 3:
            raise ValueError(
                f"Patch movie must be TYX, got ndim={video.ndim}: {p.movie_path}"
            )
        if int(video.shape[0]) < global_t:
            raise ValueError(
                f"Patch movie shorter than required global_t for {p.patch_name}: {video.shape[0]} < {global_t}"
            )

        for start in range(0, global_t, model_chunk):
            end = min(start + model_chunk, global_t)
            tc = end - start

            ac_mat: np.ndarray | None = None
            if ("AC" in video_outputs) or need_background:
                c_chunk = C[:, start:end].astype(np.float32, copy=False)
                ac_mat = np.asarray(A @ c_chunk, dtype=np.float32)
                if "AC" in video_outputs:
                    ac_frames = ac_mat.T.reshape((tc, p.h, p.w), order="F")
                    video_outputs["AC"][
                        start:end,
                        p.y0 : p.y1,
                        p.x0 : p.x1,
                    ] = _crop_core_3d(p, ac_frames)

            if "AC-qc" in video_outputs:
                if A_qc is None or C_qc is None:
                    ac_qc_frames = np.zeros((tc, p.h, p.w), dtype=np.float32)
                else:
                    c_qc_chunk = C_qc[:, start:end].astype(np.float32, copy=False)
                    ac_qc_mat = np.asarray(A_qc @ c_qc_chunk, dtype=np.float32)
                    ac_qc_frames = ac_qc_mat.T.reshape((tc, p.h, p.w), order="F")
                video_outputs["AC-qc"][
                    start:end,
                    p.y0 : p.y1,
                    p.x0 : p.x1,
                ] = _crop_core_3d(p, ac_qc_frames)

            if "ACYrA" in video_outputs:
                if YrA is None:
                    raise RuntimeError(f"Missing YrA while writing ACYrA for {p.patch_name}")
                acyra_chunk = C[:, start:end] + YrA[:, start:end]
                acyra_mat = np.asarray(A @ acyra_chunk, dtype=np.float32)
                acyra_frames = acyra_mat.T.reshape((tc, p.h, p.w), order="F")
                video_outputs["ACYrA"][
                    start:end,
                    p.y0 : p.y1,
                    p.x0 : p.x1,
                ] = _crop_core_3d(p, acyra_frames)

            if "ACYrA-qc" in video_outputs:
                if A_qc is None or C_qc is None or YrA_qc is None:
                    acyra_qc_frames = np.zeros((tc, p.h, p.w), dtype=np.float32)
                else:
                    acyra_qc_chunk = C_qc[:, start:end] + YrA_qc[:, start:end]
                    acyra_qc_mat = np.asarray(A_qc @ acyra_qc_chunk, dtype=np.float32)
                    acyra_qc_frames = acyra_qc_mat.T.reshape((tc, p.h, p.w), order="F")
                video_outputs["ACYrA-qc"][
                    start:end,
                    p.y0 : p.y1,
                    p.x0 : p.x1,
                ] = _crop_core_3d(p, acyra_qc_frames)

            if need_background:
                if ac_mat is None:
                    raise RuntimeError(f"Missing AC while writing background for {p.patch_name}")
                if W is None or bc_flat is None or bc_core is None:
                    raise RuntimeError(f"Missing background inputs for {p.patch_name}")
                block = np.asarray(video[start:end], dtype=np.float32)
                y_mat = np.transpose(block, (1, 2, 0)).reshape(
                    (p.h * p.w, tc),
                    order="F",
                )
                residual = y_mat - ac_mat - bc_flat[:, None]
                bf_mat = np.asarray(W @ residual, dtype=np.float32)
                bf_frames = bf_mat.T.reshape((tc, p.h, p.w), order="F")
                bf_core = _crop_core_3d(p, bf_frames)
                if "Bf" in video_outputs:
                    video_outputs["Bf"][
                        start:end,
                        p.y0 : p.y1,
                        p.x0 : p.x1,
                    ] = bf_core
                if "B" in video_outputs:
                    video_outputs["B"][
                        start:end,
                        p.y0 : p.y1,
                        p.x0 : p.x1,
                    ] = bf_core + bc_core[None, :, :]
def save_model_products_stitched(
    y_load_path: Path,
    extract_fold_cfg: str | None,
    core_specs: dict[str, tuple[int, int, int, int]],
    save_dir: Path,
    t_snr: Sequence[float | None] | None,
    t_r_value: Sequence[float | None] | None,
    t_lam: Sequence[float | None] | None,
    t_neurons_sn: Sequence[float | None] | None,
    requested_output_names: Sequence[str] | None = None,
) -> None:
    y_load_fold, extract_fold = _resolve_patch_folders(
        y_load_path,
        extract_fold_cfg,
    )
    extract_root = _resolve_extract_root(extract_fold)
    profile_dir = extract_root / "patch-profile"
    qc_threshold_specs = _load_qc_threshold_specs(
        extract_root=extract_root,
        t_snr=t_snr,
        t_r_value=t_r_value,
        t_lam=t_lam,
        t_neurons_sn=t_neurons_sn,
    )
    patches = _collect_patch_meta(y_load_fold, extract_fold)
    if not core_specs:
        raise ValueError("core_specs is empty; cannot stitch model products.")
    full_h = max(v[1] for v in core_specs.values())
    full_w = max(v[3] for v in core_specs.values())
    if full_h <= 0 or full_w <= 0:
        raise ValueError(f"Invalid stitched core shape from core_specs: {(full_h, full_w)}")
    for p in patches:
        if p.patch_name not in core_specs:
            raise ValueError(f"Missing core patch spec for model patch: {p.patch_name}")
        y0, y1, x0, x1 = core_specs[p.patch_name]
        core_h = int(y1 - y0)
        core_w = int(x1 - x0)
        if core_h <= 0 or core_w <= 0:
            raise ValueError(
                f"Invalid core span for {p.patch_name}: y=({y0},{y1}) x=({x0},{x1})"
            )
        pad_h = int(p.h - core_h)
        pad_w = int(p.w - core_w)
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Core larger than patch for {p.patch_name}: "
                f"core={(core_h, core_w)}, patch={(p.h, p.w)}"
            )
        if (pad_h % 2) != 0 or (pad_w % 2) != 0:
            raise ValueError(
                f"Cannot infer symmetric pad for {p.patch_name}: "
                f"patch={(p.h, p.w)}, core={(core_h, core_w)}"
            )
        p.y0, p.y1, p.x0, p.x1 = y0, y1, x0, x1
        p.core_h, p.core_w = core_h, core_w
        p.core_ly0 = pad_h // 2
        p.core_lx0 = pad_w // 2
        p.core_ly1 = p.core_ly0 + core_h
        p.core_lx1 = p.core_lx0 + core_w

    logging.getLogger("caiman").setLevel(logging.ERROR)
    output_paths = _model_output_path_map(save_dir)
    missing_outputs = _get_missing_model_outputs(
        save_dir=save_dir,
        requested_output_names=requested_output_names,
    )
    if not missing_outputs:
        return
    global_t = min(p.t for p in patches)
    if global_t <= 0:
        raise ValueError("No frames available from patch movies.")

    static_names = tuple(name for name in MODEL_STATIC_OUTPUT_ORDER if name in missing_outputs)
    video_names = tuple(name for name in MODEL_VIDEO_OUTPUT_ORDER if name in missing_outputs)
    qc_static_names = tuple(
        name for name in MODEL_QC_STATIC_OUTPUT_ORDER if name in missing_outputs
    )
    qc_video_names = tuple(
        name for name in MODEL_QC_VIDEO_OUTPUT_ORDER if name in missing_outputs
    )
    static_names_all = static_names + qc_static_names
    video_names_all = video_names + qc_video_names
    needs_video_zero_init = bool(video_names_all) and _grid_has_holes(patches, full_h, full_w)
    need_ac = (
        ("AC" in video_names_all)
        or ("Bf" in video_names_all)
        or ("B" in video_names_all)
        or ("ACsum-qc" in static_names_all)
        or ("AC-qc" in video_names_all)
    )
    need_acyra = (
        ("ACYrA" in video_names_all)
        or ("ACYrAsum" in static_names_all)
        or ("ACYrAsum-qc" in static_names_all)
        or ("ACYrA-qc" in video_names_all)
    )
    need_background = ("Bf" in video_names) or ("B" in video_names)
    need_qc = bool(qc_static_names or qc_video_names)
    if need_qc and not profile_dir.is_dir():
        raise FileNotFoundError(
            f"QC outputs requested but profile directory not found: {profile_dir}"
        )

    _cleanup_legacy_temp_dirs(save_dir)
    created_paths: list[Path] = []
    static_outputs: dict[str, np.memmap] = {}
    video_outputs: dict[str, np.memmap] = {}
    try:
        for name in static_names_all:
            path = output_paths[name]
            static_outputs[name] = _create_output_memmap(
                path=path,
                shape=(full_h, full_w),
                zero_init=True,
            )
            created_paths.append(path)
        for name in video_names_all:
            path = output_paths[name]
            video_outputs[name] = _create_output_memmap(
                path=path,
                shape=(global_t, full_h, full_w),
                zero_init=needs_video_zero_init,
            )
            created_paths.append(path)

        progress_desc = "save(extract)"
        if static_names_all and not video_names_all:
            progress_desc = "save(extract2D)"
        elif video_names_all and not static_names_all:
            progress_desc = "save(extract3D)"

        with tqdm.tqdm(
            total=len(patches),
            desc=progress_desc,
            unit="patch",
            dynamic_ncols=True,
        ) as patch_progress_bar:
            for p in patches:
                patch_model_chunk = _choose_model_chunk(
                    p=p,
                    global_t=global_t,
                    need_ac=need_ac,
                    need_acyra=need_acyra,
                    need_background=need_background,
                )
                _process_single_patch_into_outputs(
                    p=p,
                    global_t=global_t,
                    model_chunk=patch_model_chunk,
                    static_outputs=static_outputs,
                    video_outputs=video_outputs,
                    profile_dir=profile_dir if need_qc else None,
                    qc_threshold_specs=qc_threshold_specs,
                )
                patch_progress_bar.update(1)
        for array in static_outputs.values():
            array.flush()
        for array in video_outputs.values():
            array.flush()
    except Exception:
        static_outputs.clear()
        video_outputs.clear()
        for path in created_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def save_time_mean_std_tif(
    x_path: Path,
    mean_out_path: Path,
    std_out_path: Path,
    save_mean: bool = True,
    save_std: bool = True,
    extra_output_labels: Sequence[str] | None = None,
) -> None:
    if not save_mean and not save_std:
        return
    stats_labels = [
        label
        for enabled, label in (
            (save_mean, mean_out_path.stem),
            (save_std, std_out_path.stem),
        )
        if enabled
    ]
    if extra_output_labels is not None:
        stats_labels.extend(str(label) for label in extra_output_labels)
    stats_label = ",".join(stats_labels)

    with tifffile.TiffFile(x_path) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            frame = np.asarray(video, dtype=np.float32)
            if save_mean:
                tifffile.imwrite(
                    mean_out_path,
                    frame,
                    photometric="minisblack",
                )
            if save_std:
                tifffile.imwrite(
                    std_out_path,
                    np.zeros_like(frame, dtype=np.float32),
                    photometric="minisblack",
                )
            return

        if video.ndim != 3:
            raise ValueError(f"Expected 2D or 3D TIFF, got ndim={video.ndim}: {x_path}")

        t = int(video.shape[0])
        h = int(video.shape[1])
        w = int(video.shape[2])

        pixel_count = max(1, h * w)
        float32_bytes = np.dtype(np.float32).itemsize
        stats_chunk, can_use_single_pass = _choose_stats_chunk(
            t=t,
            h=h,
            w=w,
            save_std=save_std,
        )

        if can_use_single_pass:
            if save_mean and mean_out_path.exists():
                mean_out_path.unlink()
            if save_std and std_out_path.exists():
                std_out_path.unlink()

            mean_acc = np.zeros((h, w), dtype=np.float64)
            sumsq_acc = (
                np.zeros((h, w), dtype=np.float64)
                if save_std
                else None
            )
            total_chunks = (t + stats_chunk - 1) // stats_chunk
            for start in tqdm.tqdm(
                range(0, t, stats_chunk),
                total=total_chunks,
                desc=f"save({stats_label})",
                unit="chunk",
                dynamic_ncols=True,
            ):
                end = min(start + stats_chunk, t)
                chunk_data = np.asarray(video[start:end], dtype=np.float32)
                mean_acc += chunk_data.sum(axis=0, dtype=np.float64)
                if save_std and sumsq_acc is not None:
                    np.square(chunk_data, out=chunk_data)
                    sumsq_acc += chunk_data.sum(axis=0, dtype=np.float64)
                del chunk_data

            mean_acc /= max(t, 1)
            if save_mean:
                tifffile.imwrite(
                    mean_out_path,
                    mean_acc.astype(np.float32),
                    photometric="minisblack",
                )
            if save_std and sumsq_acc is not None:
                var_map = np.maximum(
                    sumsq_acc / max(t, 1) - np.square(mean_acc),
                    0.0,
                )
                np.sqrt(var_map, out=var_map)
                tifffile.imwrite(
                    std_out_path,
                    var_map.astype(np.float32),
                    photometric="minisblack",
                )
            return

        # Low-memory fallback: keep an automatically chosen time chunk and reduce
        # spatial working set instead of failing the whole save.
        target_chunk_bytes = 64 * 1024 * 1024
        bytes_per_row = max(1, stats_chunk * w * float32_bytes)
        row_block = max(1, min(h, target_chunk_bytes // bytes_per_row))

        if save_mean and mean_out_path.exists():
            mean_out_path.unlink()
        if save_std and std_out_path.exists():
            std_out_path.unlink()

        mean_map = (
            tifffile.memmap(
                mean_out_path,
                shape=(h, w),
                dtype=np.float32,
                photometric="minisblack",
            )
            if save_mean
            else None
        )
        std_map = (
            tifffile.memmap(
                std_out_path,
                shape=(h, w),
                dtype=np.float32,
                photometric="minisblack",
            )
            if save_std
            else None
        )

        total_blocks = (h + row_block - 1) // row_block
        for y0 in tqdm.tqdm(
            range(0, h, row_block),
            total=total_blocks,
            desc=f"save({stats_label},row)",
            unit="block",
            dynamic_ncols=True,
        ):
            y1 = min(y0 + row_block, h)
            block_sum = np.zeros((y1 - y0, w), dtype=np.float64)
            block_sumsq = (
                np.zeros((y1 - y0, w), dtype=np.float64)
                if save_std
                else None
            )

            for start in range(0, t, stats_chunk):
                end = min(start + stats_chunk, t)
                chunk_data = np.asarray(video[start:end, y0:y1, :], dtype=np.float32)
                block_sum += chunk_data.sum(axis=0, dtype=np.float64)
                if save_std and block_sumsq is not None:
                    np.square(chunk_data, out=chunk_data)
                    block_sumsq += chunk_data.sum(axis=0, dtype=np.float64)
                del chunk_data

            block_mean = block_sum / max(t, 1)
            if save_mean and mean_map is not None:
                mean_map[y0:y1, :] = block_mean.astype(np.float32, copy=False)
            if save_std and std_map is not None and block_sumsq is not None:
                block_var = np.maximum(
                    block_sumsq / max(t, 1) - np.square(block_mean),
                    0.0,
                )
                np.sqrt(block_var, out=block_var)
                std_map[y0:y1, :] = block_var.astype(np.float32, copy=False)

        if mean_map is not None:
            mean_map.flush()
        if std_map is not None:
            std_map.flush()


def _is_complete_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _cleanup_legacy_temp_dirs(save_dir: Path) -> None:
    for name in LEGACY_TEMP_OUTPUT_DIRS:
        path = save_dir / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _crop_core_2d(p: PatchMeta, frame: np.ndarray) -> np.ndarray:
    return frame[p.core_ly0 : p.core_ly1, p.core_lx0 : p.core_lx1]


def _crop_core_3d(p: PatchMeta, frames: np.ndarray) -> np.ndarray:
    return frames[:, p.core_ly0 : p.core_ly1, p.core_lx0 : p.core_lx1]


def _model_output_path_map(save_dir: Path) -> dict[str, Path]:
    return {
        "Bc": save_dir / "Bc.tif",
        "ACsum": save_dir / "ACsum.tif",
        "ACYrAsum": save_dir / "ACYrAsum.tif",
        "ACsum-qc": save_dir / "ACsum-qc.tif",
        "ACYrAsum-qc": save_dir / "ACYrAsum-qc.tif",
        "Bf": save_dir / "Bf.tif",
        "B": save_dir / "B.tif",
        "AC": save_dir / "AC.tif",
        "ACYrA": save_dir / "ACYrA.tif",
        "AC-qc": save_dir / "AC-qc.tif",
        "ACYrA-qc": save_dir / "ACYrA-qc.tif",
    }


def _get_missing_model_outputs(
    save_dir: Path,
    requested_output_names: Sequence[str] | None = None,
) -> dict[str, Path]:
    output_paths = _model_output_path_map(save_dir)
    requested_names = (
        MODEL_OUTPUT_ORDER
        if requested_output_names is None
        else tuple(requested_output_names)
    )
    unknown_outputs = [name for name in requested_names if name not in output_paths]
    if unknown_outputs:
        raise ValueError(f"Unknown model outputs requested: {unknown_outputs}")
    return {
        name: output_paths[name]
        for name in requested_names
        if not _is_complete_file(output_paths[name])
    }


def _format_model_bar_desc(name: str, direction: str) -> str:
    return f"save({name.rjust(MODEL_BAR_LABEL_WIDTH)}{direction}out)"


def run(
    x_load_path: str,
    y_load_path: str,
    extract_fold: str | None,
    save_fold: str | None,
    crop_load_fold: str | None,
    t_snr: Sequence[float | None] | None,
    t_r_value: Sequence[float | None] | None,
    t_lam: Sequence[float | None] | None,
    t_neurons_sn: Sequence[float | None] | None,
    extract2d: bool = True,
    extract3d: bool = True,
) -> None:
    input_path = Path(x_load_path)
    y_input_path = Path(y_load_path)
    plot_dir = (
        Path(save_fold)
        if save_fold is not None
        else y_input_path.parent
    )
    x_base = input_path.with_suffix("").name
    y_base = y_input_path.with_suffix("").name
    save_path_mean_x = plot_dir / f"{x_base}mean.tif"
    save_path_std_x = plot_dir / f"{x_base}std.tif"
    save_path_bandpass_x = plot_dir / f"{x_base}bandpass.tif"
    save_path_mean_y = plot_dir / f"{y_base}mean.tif"
    save_path_std_y = plot_dir / f"{y_base}std.tif"
    save_path_bandpass_y = plot_dir / f"{y_base}bandpass.tif"

    plot_dir.mkdir(parents=True, exist_ok=True)
    resolved_extract_fold = (
        y_input_path.parent / "extract"
        if extract_fold is None
        else Path(extract_fold)
    )
    should_save_model_products = resolved_extract_fold.is_dir()

    need_mean_x = not _is_complete_file(save_path_mean_x)
    need_std_x = not _is_complete_file(save_path_std_x)
    need_bandpass_x = not _is_complete_file(save_path_bandpass_x)
    if need_mean_x or need_std_x:
        save_time_mean_std_tif(
            x_path=input_path,
            mean_out_path=save_path_mean_x,
            std_out_path=save_path_std_x,
            save_mean=need_mean_x,
            save_std=need_std_x,
            extra_output_labels=(save_path_bandpass_x.stem,) if need_bandpass_x else None,
        )
    if not _is_complete_file(save_path_bandpass_x):
        if not _is_complete_file(save_path_std_x):
            raise FileNotFoundError(
                f"Cannot save bandpass image because std TIFF is missing or incomplete: {save_path_std_x}"
            )
        save_imagej_bandpass_tif(
            input_path=save_path_std_x,
            output_path=save_path_bandpass_x,
            progress_label=None if (need_mean_x or need_std_x) else save_path_bandpass_x.stem,
        )

    need_mean_y = not _is_complete_file(save_path_mean_y)
    need_std_y = not _is_complete_file(save_path_std_y)
    need_bandpass_y = not _is_complete_file(save_path_bandpass_y)
    if need_mean_y or need_std_y:
        save_time_mean_std_tif(
            x_path=y_input_path,
            mean_out_path=save_path_mean_y,
            std_out_path=save_path_std_y,
            save_mean=need_mean_y,
            save_std=need_std_y,
            extra_output_labels=(save_path_bandpass_y.stem,) if need_bandpass_y else None,
        )
    if not _is_complete_file(save_path_bandpass_y):
        if not _is_complete_file(save_path_std_y):
            raise FileNotFoundError(
                f"Cannot save bandpass image because std TIFF is missing or incomplete: {save_path_std_y}"
            )
        save_imagej_bandpass_tif(
            input_path=save_path_std_y,
            output_path=save_path_bandpass_y,
            progress_label=None if (need_mean_y or need_std_y) else save_path_bandpass_y.stem,
        )

    core_specs: dict[str, tuple[int, int, int, int]] | None = None
    if should_save_model_products:
        extract_root = _resolve_extract_root(resolved_extract_fold)
        requested_2d_output_names = MODEL_2D_OUTPUT_ORDER if extract2d else ()
        requested_3d_output_names = MODEL_3D_OUTPUT_ORDER if extract3d else ()
        missing_2d_outputs = _get_missing_model_outputs(
            save_dir=plot_dir,
            requested_output_names=requested_2d_output_names,
        )
        missing_3d_outputs = _get_missing_model_outputs(
            save_dir=plot_dir,
            requested_output_names=requested_3d_output_names,
        )

        if missing_2d_outputs or missing_3d_outputs:
            if crop_load_fold is None:
                raise ValueError("save requires crop_load_fold pointing to the crop folder.")
            if core_specs is None:
                crop_params = load_crop_params(Path(crop_load_fold))
                patch_specs = crop_params["patch_specs"]
                core_specs = {
                    spec.name: (int(spec.y0), int(spec.y1), int(spec.x0), int(spec.x1))
                    for spec in patch_specs
                }

            if missing_2d_outputs:
                save_model_products_stitched(
                    y_load_path=y_input_path,
                    extract_fold_cfg=str(resolved_extract_fold),
                    core_specs=core_specs,
                    save_dir=plot_dir,
                    t_snr=t_snr,
                    t_r_value=t_r_value,
                    t_lam=t_lam,
                    t_neurons_sn=t_neurons_sn,
                    requested_output_names=requested_2d_output_names,
                )

            if missing_3d_outputs:
                save_model_products_stitched(
                    y_load_path=y_input_path,
                    extract_fold_cfg=str(resolved_extract_fold),
                    core_specs=core_specs,
                    save_dir=plot_dir,
                    t_snr=t_snr,
                    t_r_value=t_r_value,
                    t_lam=t_lam,
                    t_neurons_sn=t_neurons_sn,
                    requested_output_names=requested_3d_output_names,
                )

        qc_counts = _summarize_qc_component_counts(
            extract_root=extract_root,
            t_snr=t_snr,
            t_r_value=t_r_value,
            t_lam=t_lam,
            t_neurons_sn=t_neurons_sn,
        )
        save_overlap_outputs(
            extract_root=extract_root,
            background_path=save_path_bandpass_y,
        )
        if qc_counts is not None:
            before_qc, after_qc = qc_counts
            print(f"beforeQC: {before_qc}, afterQC: {after_qc}")


class Save:
    def __init__(
        self,
        x_load_path: str,
        y_load_path: str,
        extract_fold: str | None,
        save_fold: str | None,
        crop_load_fold: str | None,
        t_snr: Sequence[float | None] | None,
        t_r_value: Sequence[float | None] | None,
        t_lam: Sequence[float | None] | None,
        t_neurons_sn: Sequence[float | None] | None,
        extract2d: bool = True,
        extract3d: bool = True,
        enable: bool = True,
        **kwargs,
    ) -> None:
        if kwargs:
            unknown_keys = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected save config keys: {unknown_keys}")
        self.x_load_path = x_load_path
        self.y_load_path = y_load_path
        self.extract_fold = extract_fold
        self.save_fold = save_fold
        self.crop_load_fold = crop_load_fold
        self.t_snr = _normalize_qc_bounds(t_snr, "snr")
        self.t_r_value = _normalize_qc_bounds(t_r_value, "r_value")
        self.t_lam = _normalize_qc_bounds(t_lam, "lam")
        self.t_neurons_sn = _normalize_qc_bounds(t_neurons_sn, "neurons_sn")
        self.extract2d = bool(extract2d)
        self.extract3d = bool(extract3d)

    def forward(self) -> None:
        run(
            x_load_path=self.x_load_path,
            y_load_path=self.y_load_path,
            extract_fold=self.extract_fold,
            save_fold=self.save_fold,
            crop_load_fold=self.crop_load_fold,
            t_snr=self.t_snr,
            t_r_value=self.t_r_value,
            t_lam=self.t_lam,
            t_neurons_sn=self.t_neurons_sn,
            extract2d=self.extract2d,
            extract3d=self.extract3d,
        )
