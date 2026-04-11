from __future__ import annotations

import argparse
import gc
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter, shift as nd_shift
from scipy.optimize import least_squares
from skimage.filters import frangi
from tqdm import tqdm


Roi = tuple[int, int, int, int]
FitWindow = tuple[int, int, int, int]
FitPoint = tuple[int, int]

DEFAULT_FIT_POINT_RADIUS = 20
DEFAULT_VESSEL_KEEP_PERCENT = 18.0
DEFAULT_FRANGI_SIGMAS = (3.0, 4.0, 5.0, 6.0)
DEFAULT_CHUNK_SIZE = 32
ROI_BOX_COLOR = "yellow"
INTEGER_PEAK_COLOR = "cyan"
FIT_COLOR = "magenta"
FIT_WINDOW_COLOR = ROI_BOX_COLOR


class MatchPointStitch:
    def __init__(
        self,
        match_point_1: tuple[tuple[int, int], tuple[int, int]],
        match_point_2: tuple[tuple[int, int], tuple[int, int]],
        match_point_3: tuple[tuple[int, int], tuple[int, int]],
        match_brightness: bool = False,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        self.match_point_1 = np.asarray(match_point_1, dtype=np.int64)
        self.match_point_2 = np.asarray(match_point_2, dtype=np.int64)
        self.match_point_3 = np.asarray(match_point_3, dtype=np.int64)
        self.match_brightness = bool(match_brightness)
        self.dtype = np.dtype(dtype)

    def forward(self, frames: np.ndarray) -> np.ndarray:
        chunk, is_single = self._to_chunk(frames)
        stitched = np.stack([self._stitch_frame(chunk[i]) for i in range(chunk.shape[0])], axis=0).astype(self.dtype, copy=False)
        return stitched[0] if is_single else stitched

    def _stitch_frame(self, frame: np.ndarray) -> np.ndarray:
        img_raw = frame.astype(self.dtype, copy=False)
        h, w = img_raw.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("Input frame height and width must be even.")
        half_h = h // 2
        half_w = w // 2
        img_raw_sub_1 = img_raw[:half_h, half_w:]
        img_raw_sub_2 = img_raw[:half_h, :half_w]
        img_raw_sub_3 = img_raw[half_h:, :half_w]
        img_raw_sub_4 = img_raw[half_h:, half_w:]
        stitch_1 = self._manual_stitch_horizontal(np.rot90(img_raw_sub_2, 2), np.rot90(img_raw_sub_1, 2), self.match_point_1)
        stitch_2 = self._manual_stitch_horizontal(np.rot90(img_raw_sub_3, 2), np.rot90(img_raw_sub_4, 2), self.match_point_2)
        return self._manual_stitch_vertical(stitch_1, stitch_2, self.match_point_3)

    def _manual_stitch_horizontal(self, img_a: np.ndarray, img_b: np.ndarray, match_point: np.ndarray) -> np.ndarray:
        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        shift_x = int(w_a - match_point[0, 0] + match_point[1, 0])
        shift_y = int(match_point[1, 1] - match_point[0, 1])
        img_a_overlap_x = self._matlab_range(w_a - shift_x + 1, w_a)
        img_b_overlap_x = self._matlab_range(1, shift_x)
        img_a_overlap_y = self._matlab_range(max(1 - shift_y, 1), min(h_b - shift_y, h_a))
        img_b_overlap_y = self._matlab_range(max(1 + shift_y, 1), min(h_a + shift_y, h_b))
        stitched = np.zeros((img_a_overlap_y.size, w_a + w_b - img_a_overlap_x.size), dtype=self.dtype)
        stitched[:, :w_a] = self._alpha_blending(img_a[img_a_overlap_y, :], 0, 0, 0, shift_x)
        if self.match_brightness:
            denom = np.mean(img_b[np.ix_(img_b_overlap_y, img_b_overlap_x)])
            ratio = np.mean(img_a[np.ix_(img_a_overlap_y, img_a_overlap_x)]) / denom if denom != 0 else 1.0
        else:
            ratio = 1.0
        stitched[:, w_a - shift_x :] += ratio * self._alpha_blending(img_b[img_b_overlap_y, :], 0, 0, shift_x, 0)
        return stitched

    def _manual_stitch_vertical(self, img_a: np.ndarray, img_b: np.ndarray, match_point: np.ndarray) -> np.ndarray:
        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        shift_x = int(match_point[1, 0] - match_point[0, 0])
        shift_y = int(h_a - match_point[0, 1] + match_point[1, 1])
        img_a_overlap_x = self._matlab_range(max(1 - shift_x, 1), min(w_b - shift_x, w_a))
        img_b_overlap_x = self._matlab_range(max(1 + shift_x, 1), min(w_a + shift_x, w_b))
        img_a_overlap_y = self._matlab_range(h_a - shift_y + 1, h_a)
        img_b_overlap_y = self._matlab_range(1, shift_y)
        stitched = np.zeros((h_a + h_b - img_a_overlap_y.size, img_a_overlap_x.size), dtype=self.dtype)
        stitched[:h_a, :] = self._alpha_blending(img_a[:, img_a_overlap_x], 0, shift_y, 0, 0)
        if self.match_brightness:
            denom = np.mean(img_b[np.ix_(img_b_overlap_y, img_b_overlap_x)])
            ratio = np.mean(img_a[np.ix_(img_a_overlap_y, img_a_overlap_x)]) / denom if denom != 0 else 1.0
        else:
            ratio = 1.0
        stitched[h_a - shift_y :, :] += ratio * self._alpha_blending(img_b[:, img_b_overlap_x], shift_y, 0, 0, 0)
        return stitched

    @staticmethod
    def _to_chunk(frames: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(frames)
        if arr.ndim == 2:
            return arr[None, ...], True
        if arr.ndim == 3:
            return arr, False
        raise ValueError("Input must be a 2D frame or 3D chunk (T, H, W).")

    @staticmethod
    def _matlab_range(start: int, end: int) -> np.ndarray:
        if end < start:
            return np.empty((0,), dtype=np.int64)
        return np.arange(start - 1, end, dtype=np.int64)

    def _alpha_blending(self, img: np.ndarray, overlap_top: int, overlap_bottom: int, overlap_left: int, overlap_right: int) -> np.ndarray:
        m, n = img.shape
        alpha = np.ones((m, n), dtype=self.dtype)
        if overlap_top > 0:
            alpha[:overlap_top, :] = np.repeat(self._robust_linspace(0, 1, overlap_top)[:, None], n, axis=1)
        if overlap_bottom > 0:
            alpha[m - overlap_bottom :, :] = np.repeat(self._robust_linspace(1, 0, overlap_bottom)[:, None], n, axis=1)
        if overlap_left > 0:
            alpha[:, :overlap_left] *= np.repeat(self._robust_linspace(0, 1, overlap_left)[None, :], m, axis=0)
        if overlap_right > 0:
            alpha[:, n - overlap_right :] *= np.repeat(self._robust_linspace(1, 0, overlap_right)[None, :], m, axis=0)
        return img * alpha

    def _robust_linspace(self, a: float, b: float, n: int) -> np.ndarray:
        if n == 1:
            return np.array([0.5], dtype=self.dtype)
        if n < 1:
            return np.empty((0,), dtype=self.dtype)
        return np.linspace(a, b, n, dtype=self.dtype)


@dataclass
class AlignmentResult:
    top_roi: Roi
    bottom_roi: Roi
    reference_patch: np.ndarray
    moving_patch: np.ndarray
    reference_patch_processed: np.ndarray
    moving_patch_processed: np.ndarray
    correlation: np.ndarray
    correlation_fit_model: np.ndarray
    integer_peak: tuple[int, int]
    fitted_peak: tuple[float, float]
    fit_initial_center: tuple[float, float]
    fit_sigma: tuple[float, float]
    fit_initial_sigma: tuple[float, float]
    fit_sigma_axes: tuple[float, float]
    fit_angle_deg: float
    fit_rho: float
    fit_window: FitWindow
    fit_success: bool
    fit_message: str
    dy_local: float
    dx_local: float
    offset_y: float
    offset_x: float
    relative_shift_y: float
    relative_shift_x: float
    preprocessing_mode: str
    reference_threshold: float
    moving_threshold: float
    stitch_axis: str


@dataclass
class PreparedAlignment:
    top_roi: Roi
    bottom_roi: Roi
    top_patch: np.ndarray
    bottom_patch: np.ndarray
    top_crop_origin: tuple[int, int]
    bottom_crop_origin: tuple[int, int]
    corr: np.ndarray
    reference_info: dict[str, float | str | np.ndarray]
    moving_info: dict[str, float | str | np.ndarray]
    base_offset_y: float
    base_offset_x: float
    corr_for_fit: np.ndarray

def resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mapping_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _json_load(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError(f"PyYAML is required to read {path.name}") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping payload in {path}")
        return payload
    raise ValueError(f"Unsupported parameter file format: {path}")


def parse_roi(values: Sequence[int] | None) -> Roi | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("ROI must contain exactly four integers: y0 y1 x0 x1")
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def parse_fit_point(values: Sequence[int] | None) -> FitPoint | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Fit point must contain exactly two integers: y x")
    return int(values[0]), int(values[1])


def clamp_roi(y0: float, x0: float, height: float, width: float, shape: tuple[int, int], min_size: int = 8) -> Roi:
    h, w = shape
    height_i = max(int(round(height)), min_size)
    width_i = max(int(round(width)), min_size)
    height_i = min(height_i, h)
    width_i = min(width_i, w)
    y0_i = int(np.clip(round(y0), 0, max(h - height_i, 0)))
    x0_i = int(np.clip(round(x0), 0, max(w - width_i, 0)))
    return y0_i, y0_i + height_i, x0_i, x0_i + width_i


def validate_fit_window(window: FitWindow, corr_shape: tuple[int, int]) -> FitWindow:
    y0, y1, x0, x1 = window
    y_start = min(y0, y1)
    x_start = min(x0, x1)
    height = abs(y1 - y0)
    width = abs(x1 - x0)
    return clamp_roi(y_start, x_start, height, width, corr_shape, min_size=4)


def roi_height_width(roi: Roi) -> tuple[int, int]:
    return roi[1] - roi[0], roi[3] - roi[2]


def point_in_roi(x: float, y: float, roi: Roi | None) -> bool:
    if roi is None:
        return False
    return roi[2] <= x <= roi[3] and roi[0] <= y <= roi[1]


def point_inside_window(point: tuple[float, float], fit_window: FitWindow) -> bool:
    y0, y1, x0, x1 = fit_window
    return y0 <= float(point[0]) < y1 and x0 <= float(point[1]) < x1


def point_to_window(point: tuple[float, float], fit_window: FitWindow) -> tuple[float, float]:
    y0, _y1, x0, _x1 = fit_window
    return float(point[0]) - y0, float(point[1]) - x0


def mirror_roi_vertical_symmetric(roi: Roi, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Roi:
    height, width = roi_height_width(roi)
    mirrored_y0 = source_shape[0] - roi[1]
    mirrored_x0 = roi[2]
    return clamp_roi(mirrored_y0, mirrored_x0, height, width, target_shape)


def mirror_roi_horizontal_symmetric(roi: Roi, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Roi:
    height, width = roi_height_width(roi)
    mirrored_y0 = roi[0]
    mirrored_x0 = source_shape[1] - roi[3]
    return clamp_roi(mirrored_y0, mirrored_x0, height, width, target_shape)


def resolve_horizontal_mirror_rois(
    reference_roi: Roi | None,
    moving_roi: Roi | None,
    reference_shape: tuple[int, int],
    moving_shape: tuple[int, int],
) -> tuple[Roi | None, Roi | None]:
    if reference_roi is not None:
        ref_h, ref_w = roi_height_width(reference_roi)
        reference_roi = clamp_roi(reference_roi[0], reference_roi[2], ref_h, ref_w, reference_shape)
        moving_roi = mirror_roi_horizontal_symmetric(reference_roi, reference_shape, moving_shape)
        return reference_roi, moving_roi
    if moving_roi is not None:
        mov_h, mov_w = roi_height_width(moving_roi)
        moving_roi = clamp_roi(moving_roi[0], moving_roi[2], mov_h, mov_w, moving_shape)
        reference_roi = mirror_roi_horizontal_symmetric(moving_roi, moving_shape, reference_shape)
        return reference_roi, moving_roi
    return None, None


def resolve_vertical_mirror_rois(
    reference_roi: Roi | None,
    moving_roi: Roi | None,
    reference_shape: tuple[int, int],
    moving_shape: tuple[int, int],
) -> tuple[Roi | None, Roi | None]:
    if reference_roi is not None:
        ref_h, ref_w = roi_height_width(reference_roi)
        reference_roi = clamp_roi(reference_roi[0], reference_roi[2], ref_h, ref_w, reference_shape)
        moving_roi = mirror_roi_vertical_symmetric(reference_roi, reference_shape, moving_shape)
        return reference_roi, moving_roi
    if moving_roi is not None:
        mov_h, mov_w = roi_height_width(moving_roi)
        moving_roi = clamp_roi(moving_roi[0], moving_roi[2], mov_h, mov_w, moving_shape)
        reference_roi = mirror_roi_vertical_symmetric(moving_roi, moving_shape, reference_shape)
        return reference_roi, moving_roi
    return None, None


def robust_clip(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [low, high])
    return np.clip(image, lo, hi)


def percentile_view(image: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [low, high])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def minmax01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = image - float(image.min())
    scale = float(image.max())
    if scale <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return image / scale


def keep_top_response(response: np.ndarray, keep_percent: float) -> tuple[np.ndarray, float]:
    response = np.asarray(response, dtype=np.float32)
    keep_percent = float(np.clip(keep_percent, 0.0, 100.0))
    positive = response[response > 0]
    if positive.size == 0:
        return np.zeros_like(response, dtype=np.float32), 0.0
    threshold = float(np.percentile(positive, 100.0 - keep_percent))
    kept = np.where(response >= threshold, response, 0.0).astype(np.float32)
    if np.any(kept > 0):
        kept = gaussian_filter(kept, sigma=1.0)
    return kept, threshold


def preprocess_patch_for_correlation(
    image: np.ndarray,
    mode: str,
    vessel_keep_percent: float,
    frangi_sigmas: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, float | str | np.ndarray]]:
    clipped = robust_clip(image)
    threshold = 0.0
    if mode == "raw":
        feature = clipped.copy()
    elif mode == "frangi":
        scaled = minmax01(clipped)
        vesselness = frangi(scaled, sigmas=frangi_sigmas, black_ridges=True).astype(np.float32)
        vesselness = np.nan_to_num(vesselness, nan=0.0, posinf=0.0, neginf=0.0)
        feature, threshold = keep_top_response(vesselness, keep_percent=vessel_keep_percent)
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    return feature.astype(np.float32), {
        "mode": mode,
        "threshold": threshold,
        "raw_patch": clipped.astype(np.float32, copy=False),
        "feature_patch": feature.astype(np.float32, copy=False),
    }


def normalize_for_phase_corr(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    feature = feature - feature.mean()
    std = feature.std()
    if std > 0:
        feature = feature / std
    wy = np.hanning(feature.shape[0])[:, None]
    wx = np.hanning(feature.shape[1])[None, :]
    return feature * wy * wx


def compute_phase_correlation_map(
    reference: np.ndarray,
    moving: np.ndarray,
    preprocessing_mode: str,
    vessel_keep_percent: float,
    frangi_sigmas: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, float | str | np.ndarray], dict[str, float | str | np.ndarray], np.ndarray]:
    if reference.shape != moving.shape:
        raise ValueError(f"Expected same shape, got {reference.shape} vs {moving.shape}")

    reference_feature, reference_info = preprocess_patch_for_correlation(
        reference,
        mode=preprocessing_mode,
        vessel_keep_percent=vessel_keep_percent,
        frangi_sigmas=frangi_sigmas,
    )
    moving_feature, moving_info = preprocess_patch_for_correlation(
        moving,
        mode=preprocessing_mode,
        vessel_keep_percent=vessel_keep_percent,
        frangi_sigmas=frangi_sigmas,
    )

    ref = normalize_for_phase_corr(reference_feature)
    mov = normalize_for_phase_corr(moving_feature)
    cps = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cps /= np.maximum(np.abs(cps), 1e-8)
    corr = np.fft.ifft2(cps)
    corr = np.fft.fftshift(np.abs(corr)).astype(np.float32, copy=False)
    return corr, reference_info, moving_info, corr.astype(np.float64, copy=False)


def center_crop_to_common_shape(
    reference_patch: np.ndarray,
    moving_patch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    common_h = min(reference_patch.shape[0], moving_patch.shape[0])
    common_w = min(reference_patch.shape[1], moving_patch.shape[1])
    ref_y0 = max((reference_patch.shape[0] - common_h) // 2, 0)
    ref_x0 = max((reference_patch.shape[1] - common_w) // 2, 0)
    mov_y0 = max((moving_patch.shape[0] - common_h) // 2, 0)
    mov_x0 = max((moving_patch.shape[1] - common_w) // 2, 0)
    reference_cropped = reference_patch[ref_y0 : ref_y0 + common_h, ref_x0 : ref_x0 + common_w]
    moving_cropped = moving_patch[mov_y0 : mov_y0 + common_h, mov_x0 : mov_x0 + common_w]
    return reference_cropped, moving_cropped, (ref_y0, ref_x0), (mov_y0, mov_x0)


def gaussian_2d_cov_with_offset(
    yy: np.ndarray,
    xx: np.ndarray,
    amplitude: float,
    center_y: float,
    center_x: float,
    log_sigma_y: float,
    log_sigma_x: float,
    rho_raw: float,
    offset: float,
) -> np.ndarray:
    sigma_y = max(float(np.exp(log_sigma_y)), 1e-6)
    sigma_x = max(float(np.exp(log_sigma_x)), 1e-6)
    rho = 0.98 * np.tanh(float(rho_raw))
    dy = yy - center_y
    dx = xx - center_x
    denom = max(1.0 - rho * rho, 1e-6)
    z = ((dy / sigma_y) ** 2 - 2.0 * rho * dy * dx / (sigma_y * sigma_x) + (dx / sigma_x) ** 2) / denom
    return offset + amplitude * np.exp(-0.5 * z)


def point_fit_window(point: tuple[float, float], shape: tuple[int, int], radius: int = DEFAULT_FIT_POINT_RADIUS) -> FitWindow:
    radius = max(int(radius), 1)
    point_y = int(round(point[0]))
    point_x = int(round(point[1]))
    y0 = max(point_y - radius, 0)
    y1 = min(point_y + radius + 1, shape[0])
    x0 = max(point_x - radius, 0)
    x1 = min(point_x + radius + 1, shape[1])
    return y0, y1, x0, x1


def fit_gaussian_in_window(
    corr: np.ndarray,
    fit_window: FitWindow,
) -> dict[str, float | bool | str | np.ndarray | tuple[int, int] | FitWindow]:
    corr = np.asarray(corr, dtype=np.float64)
    integer_peak = tuple(int(v) for v in np.unravel_index(np.argmax(corr), corr.shape))
    y0, y1, x0, x1 = fit_window
    local_corr = corr[y0:y1, x0:x1]
    yy_local, xx_local = np.indices(local_corr.shape, dtype=np.float64)
    yy_global = yy_local + y0
    xx_global = xx_local + x0

    baseline0 = float(np.percentile(local_corr, 10.0))
    amplitude0 = max(float(local_corr.max()) - baseline0, np.finfo(np.float64).eps)
    center_y0 = float((y0 + y1 - 1) / 2.0)
    center_x0 = float((x0 + x1 - 1) / 2.0)
    sigma_y0 = max(local_corr.shape[0] / 4.0, 1.5)
    sigma_x0 = max(local_corr.shape[1] / 4.0, 1.5)

    initial = np.array(
        [
            amplitude0,
            center_y0,
            center_x0,
            float(np.log(sigma_y0)),
            float(np.log(sigma_x0)),
            0.0,
            baseline0,
        ],
        dtype=np.float64,
    )
    local_min = float(local_corr.min())
    local_max = float(local_corr.max())
    lower = np.array(
        [0.0, float(y0), float(x0), float(np.log(0.5)), float(np.log(0.5)), -4.0, local_min],
        dtype=np.float64,
    )
    upper = np.array(
        [
            max(local_max * 5.0, 1.0),
            float(y1 - 1),
            float(x1 - 1),
            float(np.log(max(local_corr.shape[0], 2))),
            float(np.log(max(local_corr.shape[1], 2))),
            4.0,
            max(local_max, local_min + 1e-6),
        ],
        dtype=np.float64,
    )
    normalized = local_corr / max(local_max, np.finfo(np.float64).eps)
    weights = 0.25 + normalized**2

    def residuals(params: np.ndarray) -> np.ndarray:
        model = gaussian_2d_cov_with_offset(yy_global, xx_global, *params)
        return ((model - local_corr) * weights).ravel()

    try:
        fit = least_squares(residuals, initial, bounds=(lower, upper), method="trf", max_nfev=400)
        params = fit.x
        success = bool(fit.success)
        message = str(fit.message)
    except Exception as exc:
        params = initial
        success = False
        message = f"gaussian fit failed: {exc}"

    sigma_y = float(np.exp(params[3]))
    sigma_x = float(np.exp(params[4]))
    rho = float(0.98 * np.tanh(params[5]))
    cov = np.array(
        [
            [sigma_y * sigma_y, rho * sigma_y * sigma_x],
            [rho * sigma_y * sigma_x, sigma_x * sigma_x],
        ],
        dtype=np.float64,
    )
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    sigma_major = float(np.sqrt(max(eigvals[0], 1e-12)))
    sigma_minor = float(np.sqrt(max(eigvals[1], 1e-12)))
    major_vec = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major_vec[0], major_vec[1])))

    yy_full, xx_full = np.indices(corr.shape, dtype=np.float64)
    model = gaussian_2d_cov_with_offset(yy_full, xx_full, *params).astype(np.float32)
    return {
        "success": success,
        "message": message,
        "peak_integer": integer_peak,
        "fit_window": (y0, y1, x0, x1),
        "initial_center_y": center_y0,
        "initial_center_x": center_x0,
        "initial_sigma_y": sigma_y0,
        "initial_sigma_x": sigma_x0,
        "center_y": float(params[1]),
        "center_x": float(params[2]),
        "sigma_y": sigma_y,
        "sigma_x": sigma_x,
        "rho": rho,
        "sigma_major": sigma_major,
        "sigma_minor": sigma_minor,
        "angle_deg": angle_deg,
        "model": model,
    }


def prepare_alignment_from_manual_rois(
    top_image: np.ndarray,
    bottom_image: np.ndarray,
    top_roi: Roi,
    bottom_roi: Roi,
    preprocessing_mode: str,
    vessel_keep_percent: float,
    frangi_sigmas: tuple[float, ...],
) -> PreparedAlignment:
    top_patch_full = top_image[top_roi[0] : top_roi[1], top_roi[2] : top_roi[3]]
    bottom_patch_full = bottom_image[bottom_roi[0] : bottom_roi[1], bottom_roi[2] : bottom_roi[3]]
    top_patch, bottom_patch, top_crop_origin, bottom_crop_origin = center_crop_to_common_shape(top_patch_full, bottom_patch_full)
    base_offset_y = (top_roi[0] + top_crop_origin[0]) - (bottom_roi[0] + bottom_crop_origin[0])
    base_offset_x = (top_roi[2] + top_crop_origin[1]) - (bottom_roi[2] + bottom_crop_origin[1])
    corr, reference_info, moving_info, corr_for_fit = compute_phase_correlation_map(
        top_patch,
        bottom_patch,
        preprocessing_mode=preprocessing_mode,
        vessel_keep_percent=vessel_keep_percent,
        frangi_sigmas=frangi_sigmas,
    )
    return PreparedAlignment(
        top_roi=top_roi,
        bottom_roi=bottom_roi,
        top_patch=top_patch,
        bottom_patch=bottom_patch,
        top_crop_origin=top_crop_origin,
        bottom_crop_origin=bottom_crop_origin,
        corr=corr,
        reference_info=reference_info,
        moving_info=moving_info,
        base_offset_y=base_offset_y,
        base_offset_x=base_offset_x,
        corr_for_fit=corr_for_fit,
    )


def finalize_alignment_with_fit_window(
    prepared: PreparedAlignment,
    selected_fit_window: FitWindow,
    preprocessing_mode: str,
    reference_image_shape: tuple[int, int],
    stitch_axis: str,
) -> AlignmentResult:
    selected_fit_window = validate_fit_window(selected_fit_window, prepared.corr.shape)
    fit_info = fit_gaussian_in_window(prepared.corr_for_fit, fit_window=selected_fit_window)
    center = np.array(prepared.corr.shape) // 2
    dy_local = float(fit_info["center_y"] - center[0])
    dx_local = float(fit_info["center_x"] - center[1])
    offset_y = prepared.base_offset_y + dy_local
    offset_x = prepared.base_offset_x + dx_local
    if stitch_axis == "vertical":
        relative_shift_y = offset_y - reference_image_shape[0]
        relative_shift_x = offset_x
    elif stitch_axis == "horizontal":
        relative_shift_y = offset_y
        relative_shift_x = offset_x - reference_image_shape[1]
    else:
        raise ValueError(f"Unsupported stitch_axis: {stitch_axis}")

    return AlignmentResult(
        top_roi=prepared.top_roi,
        bottom_roi=prepared.bottom_roi,
        reference_patch=prepared.top_patch,
        moving_patch=prepared.bottom_patch,
        reference_patch_processed=np.asarray(prepared.reference_info["feature_patch"], dtype=np.float32),
        moving_patch_processed=np.asarray(prepared.moving_info["feature_patch"], dtype=np.float32),
        correlation=prepared.corr,
        correlation_fit_model=np.asarray(fit_info["model"], dtype=np.float32),
        integer_peak=tuple(int(v) for v in fit_info["peak_integer"]),
        fitted_peak=(float(fit_info["center_y"]), float(fit_info["center_x"])),
        fit_initial_center=(float(fit_info["initial_center_y"]), float(fit_info["initial_center_x"])),
        fit_sigma=(float(fit_info["sigma_y"]), float(fit_info["sigma_x"])),
        fit_initial_sigma=(float(fit_info["initial_sigma_y"]), float(fit_info["initial_sigma_x"])),
        fit_sigma_axes=(float(fit_info["sigma_major"]), float(fit_info["sigma_minor"])),
        fit_angle_deg=float(fit_info["angle_deg"]),
        fit_rho=float(fit_info["rho"]),
        fit_window=tuple(int(v) for v in fit_info["fit_window"]),
        fit_success=bool(fit_info["success"]),
        fit_message=str(fit_info["message"]),
        dy_local=dy_local,
        dx_local=dx_local,
        offset_y=offset_y,
        offset_x=offset_x,
        relative_shift_y=relative_shift_y,
        relative_shift_x=relative_shift_x,
        preprocessing_mode=preprocessing_mode,
        reference_threshold=float(prepared.reference_info["threshold"]),
        moving_threshold=float(prepared.moving_info["threshold"]),
        stitch_axis=stitch_axis,
    )


def robust_linspace(a: float, b: float, n: int) -> np.ndarray:
    if n == 1:
        return np.array([0.5], dtype=np.float32)
    if n < 1:
        return np.empty((0,), dtype=np.float32)
    return np.linspace(a, b, n, dtype=np.float32)


def _split_offset(offset: float) -> tuple[int, float]:
    base = int(math.floor(float(offset)))
    frac = float(offset) - float(base)
    return base, frac


def _shift_subpixel(image: np.ndarray, shift_y: float, shift_x: float) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    shifted = nd_shift(
        image,
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    support = nd_shift(
        np.ones_like(image, dtype=np.float32),
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    return shifted, support


def _canvas_bounds(reference_shape: tuple[int, int], moving_shape: tuple[int, int], offset_y: float, offset_x: float) -> tuple[int, int, int, int, int, int]:
    iy, _fy = _split_offset(offset_y)
    ix, _fx = _split_offset(offset_x)
    y0 = min(0, iy)
    x0 = min(0, ix)
    y1 = max(reference_shape[0], iy + moving_shape[0])
    x1 = max(reference_shape[1], ix + moving_shape[1])
    return y0, y1, x0, x1, iy, ix


def stitch_pair(reference: np.ndarray, moving: np.ndarray, offset_y: float, offset_x: float) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    y0, y1, x0, x1, iy, ix = _canvas_bounds(reference.shape, moving.shape, offset_y, offset_x)
    _iy, fy = _split_offset(offset_y)
    _ix, fx = _split_offset(offset_x)
    moving_shifted, moving_support = _shift_subpixel(moving, fy, fx)

    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    weight = np.zeros_like(canvas)
    reference_y = -y0
    reference_x = -x0
    moving_y = iy - y0
    moving_x = ix - x0

    canvas[reference_y : reference_y + reference.shape[0], reference_x : reference_x + reference.shape[1]] += reference
    weight[reference_y : reference_y + reference.shape[0], reference_x : reference_x + reference.shape[1]] += 1.0
    canvas[moving_y : moving_y + moving_shifted.shape[0], moving_x : moving_x + moving_shifted.shape[1]] += moving_shifted
    weight[moving_y : moving_y + moving_support.shape[0], moving_x : moving_x + moving_support.shape[1]] += moving_support
    return canvas / np.maximum(weight, 1e-6)


def stitch_pair_transition(
    reference: np.ndarray,
    moving: np.ndarray,
    offset_y: float,
    offset_x: float,
    *,
    blend_axis: str,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    y0, y1, x0, x1, iy, ix = _canvas_bounds(reference.shape, moving.shape, offset_y, offset_x)
    _iy, fy = _split_offset(offset_y)
    _ix, fx = _split_offset(offset_x)
    moving_shifted, moving_support = _shift_subpixel(moving, fy, fx)

    out_h = y1 - y0
    out_w = x1 - x0
    reference_y = -y0
    reference_x = -x0
    moving_y = iy - y0
    moving_x = ix - x0

    ref_data = np.zeros((out_h, out_w), dtype=np.float32)
    ref_support = np.zeros_like(ref_data)
    mov_data = np.zeros_like(ref_data)
    mov_support_map = np.zeros_like(ref_data)

    ref_data[reference_y : reference_y + reference.shape[0], reference_x : reference_x + reference.shape[1]] = reference
    ref_support[reference_y : reference_y + reference.shape[0], reference_x : reference_x + reference.shape[1]] = 1.0
    mov_data[moving_y : moving_y + moving_shifted.shape[0], moving_x : moving_x + moving_shifted.shape[1]] = moving_shifted
    mov_support_map[moving_y : moving_y + moving_support.shape[0], moving_x : moving_x + moving_support.shape[1]] = moving_support

    overlap_mask = (ref_support > 1e-6) & (mov_support_map > 1e-6)
    if not np.any(overlap_mask):
        denom = np.maximum(ref_support + mov_support_map, 1e-6)
        return (ref_data + mov_data) / denom

    ys, xs = np.nonzero(overlap_mask)
    oy0, oy1 = int(ys.min()), int(ys.max()) + 1
    ox0, ox1 = int(xs.min()), int(xs.max()) + 1
    alpha = np.ones((out_h, out_w), dtype=np.float32)
    beta = np.ones((out_h, out_w), dtype=np.float32)
    if blend_axis == "vertical":
        ramp = robust_linspace(1.0, 0.0, oy1 - oy0)[:, None]
        alpha[oy0:oy1, ox0:ox1] = np.repeat(ramp, ox1 - ox0, axis=1)
        beta[oy0:oy1, ox0:ox1] = 1.0 - alpha[oy0:oy1, ox0:ox1]
    elif blend_axis == "horizontal":
        ramp = robust_linspace(1.0, 0.0, ox1 - ox0)[None, :]
        alpha[oy0:oy1, ox0:ox1] = np.repeat(ramp, oy1 - oy0, axis=0)
        beta[oy0:oy1, ox0:ox1] = 1.0 - alpha[oy0:oy1, ox0:ox1]
    else:
        raise ValueError(f"Unsupported blend_axis: {blend_axis}")

    output = np.zeros((out_h, out_w), dtype=np.float32)
    no_overlap = ~overlap_mask
    output[no_overlap] = (ref_data + mov_data)[no_overlap] / np.maximum((ref_support + mov_support_map)[no_overlap], 1e-6)
    overlap_den = np.maximum(ref_support * alpha + mov_support_map * beta, 1e-6)
    output[overlap_mask] = (
        ref_data[overlap_mask] * alpha[overlap_mask] + mov_data[overlap_mask] * beta[overlap_mask]
    ) / overlap_den[overlap_mask]
    return output


def split_rotate_quadrants(image: np.ndarray) -> dict[str, np.ndarray]:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Expected a 2D image.")
    h, w = image.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Projection height and width must be even.")
    hh, hw = h // 2, w // 2
    return {
        "upper_left": np.rot90(image[:hh, :hw], 2),
        "upper_right": np.rot90(image[:hh, hw:], 2),
        "lower_left": np.rot90(image[hh:, :hw], 2),
        "lower_right": np.rot90(image[hh:, hw:], 2),
    }


def stitch_full_frame_transition(
    frame: np.ndarray,
    left_result: AlignmentResult,
    right_result: AlignmentResult,
    final_result: AlignmentResult,
) -> np.ndarray:
    quadrants = split_rotate_quadrants(np.asarray(frame))
    left_stitched = stitch_pair_transition(
        quadrants["upper_left"],
        quadrants["lower_left"],
        left_result.offset_y,
        left_result.offset_x,
        blend_axis="vertical",
    )
    right_stitched = stitch_pair_transition(
        quadrants["upper_right"],
        quadrants["lower_right"],
        right_result.offset_y,
        right_result.offset_x,
        blend_axis="vertical",
    )
    return stitch_pair_transition(
        left_stitched,
        right_stitched,
        final_result.offset_y,
        final_result.offset_x,
        blend_axis="horizontal",
    ).astype(np.float32, copy=False)


def stitch_frame_from_offsets(
    frame: np.ndarray,
    *,
    left_offset_y: float,
    left_offset_x: float,
    right_offset_y: float,
    right_offset_x: float,
    final_offset_y: float,
    final_offset_x: float,
    left_crop_bounds: FitWindow | None = None,
    right_crop_bounds: FitWindow | None = None,
    final_crop_bounds: FitWindow | None = None,
) -> np.ndarray:
    quadrants = split_rotate_quadrants(np.asarray(frame))
    left_stitched = crop_image(
        stitch_pair_transition(
            quadrants["upper_left"],
            quadrants["lower_left"],
            left_offset_y,
            left_offset_x,
            blend_axis="vertical",
        ),
        left_crop_bounds,
    )
    right_stitched = crop_image(
        stitch_pair_transition(
            quadrants["upper_right"],
            quadrants["lower_right"],
            right_offset_y,
            right_offset_x,
            blend_axis="vertical",
        ),
        right_crop_bounds,
    )
    final_stitched = stitch_pair_transition(
        left_stitched,
        right_stitched,
        final_offset_y,
        final_offset_x,
        blend_axis="horizontal",
    )
    return crop_image(final_stitched, final_crop_bounds).astype(np.float32, copy=False)


def _largest_valid_rectangle(mask: np.ndarray) -> FitWindow:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D support mask, got shape {binary.shape}")
    h, w = binary.shape
    if h == 0 or w == 0 or not np.any(binary):
        raise ValueError("Support mask does not contain any valid pixels.")

    heights = np.zeros(w, dtype=np.int32)
    best_area = -1
    best_window: FitWindow | None = None
    for y in range(h):
        row = binary[y]
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            current = int(heights[x]) if x < w else 0
            while stack and current < int(heights[stack[-1]]):
                top = stack.pop()
                height = int(heights[top])
                if height <= 0:
                    continue
                left = stack[-1] + 1 if stack else 0
                right = x
                area = height * (right - left)
                if area > best_area:
                    best_area = area
                    best_window = (y - height + 1, y + 1, left, right)
            stack.append(x)
    if best_window is None:
        ys, xs = np.nonzero(binary)
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return best_window


def crop_image(image: np.ndarray, crop_bounds: FitWindow | None) -> np.ndarray:
    if crop_bounds is None:
        return np.asarray(image)
    y0, y1, x0, x1 = crop_bounds
    return np.asarray(image)[y0:y1, x0:x1]


def compute_pair_crop_bounds(
    reference_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    offset_y: float,
    offset_x: float,
    *,
    blend_axis: str,
) -> FitWindow:
    reference_support = np.ones(reference_shape, dtype=np.float32)
    moving_support = np.ones(moving_shape, dtype=np.float32)
    stitched_support = stitch_pair_transition(
        reference_support,
        moving_support,
        offset_y,
        offset_x,
        blend_axis=blend_axis,
    )
    return _largest_valid_rectangle(stitched_support > 1e-6)


def _normalize_frame(frame: Sequence[int] | None, total_frames: int) -> tuple[int, int]:
    if frame is None:
        start, end = 0, -1
    else:
        values = [int(v) for v in frame]
        if len(values) != 2:
            raise ValueError("frame must be [start, end), with end < 1 meaning full range.")
        start, end = values
    if start < 0:
        raise ValueError("frame start must be >= 0.")
    if end < 1:
        end = total_frames
    end = min(end, total_frames)
    if end <= start:
        raise ValueError(f"Invalid frame range [{start}, {end}) for {total_frames} frames.")
    return start, end


def load_or_project_sum(
    path: Path,
    frame: Sequence[int],
    chunk_size: int,
    cache_dir: Path,
    use_cache: bool = True,
) -> tuple[np.ndarray, tuple[int, int]]:
    with tifffile.TiffFile(path) as tif:
        data_shape = tif.series[0].shape
        if len(data_shape) == 2:
            projection = tif.asarray().astype(np.float32, copy=False)
            return projection, (0, 1)
        if len(data_shape) != 3:
            raise ValueError("Input TIFF must be 2D or 3D.")

        total_frames = int(data_shape[0])
        start, end = _normalize_frame(frame, total_frames)
        end_tag = "all" if end == total_frames else str(end)
        cache_path = cache_dir / f"{path.stem}_sum_{start}_{end_tag}.npy"
        if use_cache and cache_path.exists():
            return np.load(cache_path).astype(np.float32, copy=False), (start, end)

        projection = np.zeros(data_shape[1:], dtype=np.float64)
        for chunk_start in range(start, end, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, end)
            chunk = tif.asarray(key=range(chunk_start, chunk_stop))
            if chunk.ndim == 2:
                chunk = chunk[None, ...]
            projection += np.asarray(chunk, dtype=np.float64).sum(axis=0, dtype=np.float64)
            del chunk
            gc.collect()

    projection = projection.astype(np.float32)
    np.save(cache_path, projection)
    return projection, (start, end)


@contextmanager
def _noninteractive_save():
    was_interactive = plt.isinteractive()
    try:
        plt.ioff()
        yield
    finally:
        if was_interactive:
            plt.ion()


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int = 240) -> Path:
    with _noninteractive_save():
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def array_display_limits(image: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def pair_display_limits(*arrays: np.ndarray) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for arr in arrays:
        vmin, vmax = array_display_limits(arr)
        mins.append(vmin)
        maxs.append(vmax)
    return min(mins), max(maxs)


def save_linear_grayscale_png(
    path: Path,
    image: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 240,
    rectangles: list[tuple[Roi, str, float]] | None = None,
) -> Path:
    image = np.asarray(image, dtype=np.float32)
    auto_vmin, auto_vmax = array_display_limits(image)
    vmin = auto_vmin if vmin is None else float(vmin)
    vmax = auto_vmax if vmax is None else float(vmax)
    with _noninteractive_save():
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.imshow(image, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        if rectangles is not None:
            for roi, color, linewidth in rectangles:
                rect = patches.Rectangle(
                    (roi[2], roi[0]),
                    roi[3] - roi[2],
                    roi[1] - roi[0],
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                )
                ax.add_patch(rect)
        ax.axis("off")
    return _save_figure(fig, path, dpi=dpi)


def _ellipse_half_extents(sigma_major: float, sigma_minor: float, angle_deg: float, scale: float) -> tuple[float, float]:
    a = max(float(sigma_major) * float(scale), 1e-6)
    b = max(float(sigma_minor) * float(scale), 1e-6)
    theta = np.deg2rad(float(angle_deg))
    half_w = float(np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2))
    half_h = float(np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2))
    return half_h, half_w


def ellipse_fits_in_window(
    center_y: float,
    center_x: float,
    sigma_major: float,
    sigma_minor: float,
    angle_deg: float,
    shape: tuple[int, int],
    scale: float,
) -> bool:
    half_h, half_w = _ellipse_half_extents(sigma_major, sigma_minor, angle_deg, scale)
    return (
        center_y - half_h >= 0.0
        and center_y + half_h <= shape[0] - 1
        and center_x - half_w >= 0.0
        and center_x + half_w <= shape[1] - 1
    )


def correlation_input_name(mode: str) -> str:
    if mode == "raw":
        return "raw"
    if mode == "frangi":
        return "frangi"
    return mode


def build_output_naming(reference_name: str, moving_name: str, preprocessing_mode: str) -> dict[str, str]:
    if reference_name == "top" and moving_name == "bottom":
        return {
            "reference_all_raw": "T-all-raw.png",
            "moving_all_raw": "B-all-raw.png",
            "reference_roi_raw": "T-roi-raw.png",
            "moving_roi_raw": "B-roi-raw.png",
            "reference_all_processed": "T-all-vessel.png",
            "moving_all_processed": "B-all-vessel.png",
            "reference_roi_processed": "T-roi-vessel.png",
            "moving_roi_processed": "B-roi-vessel.png",
            "heatmap_all_raw": "HeatMap-all-raw.png",
            "heatmap_all_label": "HeatMap-all-label.png",
            "heatmap_roi_raw": "HeatMap-roi-raw.png",
            "heatmap_roi_label": "HeatMap-roi-label.png",
        }
    if reference_name == "left" and moving_name == "right":
        processed_suffix = "vessel" if preprocessing_mode == "frangi" else "corr"
        return {
            "reference_all_raw": "L-all-raw.png",
            "moving_all_raw": "R-all-raw.png",
            "reference_roi_raw": "L-roi-raw.png",
            "moving_roi_raw": "R-roi-raw.png",
            "reference_all_processed": f"L-all-{processed_suffix}.png",
            "moving_all_processed": f"R-all-{processed_suffix}.png",
            "reference_roi_processed": f"L-roi-{processed_suffix}.png",
            "moving_roi_processed": f"R-roi-{processed_suffix}.png",
            "heatmap_all_raw": "HeatMap-all-raw.png",
            "heatmap_all_label": "HeatMap-all-label.png",
            "heatmap_roi_raw": "HeatMap-roi-raw.png",
            "heatmap_roi_label": "HeatMap-roi-label.png",
        }
    raise ValueError(f"Unsupported naming pair: {reference_name}, {moving_name}")


def save_pair_roi_overview(output_dir: Path, reference_image: np.ndarray, moving_image: np.ndarray, result: AlignmentResult, *, naming: dict[str, str]) -> dict[str, Path]:
    raw_vmin, raw_vmax = pair_display_limits(reference_image, moving_image)
    return {
        "reference_all_raw": save_linear_grayscale_png(output_dir / naming["reference_all_raw"], reference_image, vmin=raw_vmin, vmax=raw_vmax, rectangles=[(result.top_roi, ROI_BOX_COLOR, 2.0)]),
        "moving_all_raw": save_linear_grayscale_png(output_dir / naming["moving_all_raw"], moving_image, vmin=raw_vmin, vmax=raw_vmax, rectangles=[(result.bottom_roi, ROI_BOX_COLOR, 2.0)]),
        "reference_roi_raw": save_linear_grayscale_png(output_dir / naming["reference_roi_raw"], result.reference_patch, vmin=raw_vmin, vmax=raw_vmax),
        "moving_roi_raw": save_linear_grayscale_png(output_dir / naming["moving_roi_raw"], result.moving_patch, vmin=raw_vmin, vmax=raw_vmax),
    }


def save_pair_processed_roi_overview(
    output_dir: Path,
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    result: AlignmentResult,
    vessel_keep_percent: float,
    frangi_sigmas: tuple[float, ...],
    *,
    naming: dict[str, str],
) -> dict[str, Path]:
    reference_image_processed, _ = preprocess_patch_for_correlation(reference_image, mode=result.preprocessing_mode, vessel_keep_percent=vessel_keep_percent, frangi_sigmas=frangi_sigmas)
    moving_image_processed, _ = preprocess_patch_for_correlation(moving_image, mode=result.preprocessing_mode, vessel_keep_percent=vessel_keep_percent, frangi_sigmas=frangi_sigmas)
    processed_vmin, processed_vmax = pair_display_limits(reference_image_processed, moving_image_processed)
    return {
        "reference_all_processed": save_linear_grayscale_png(output_dir / naming["reference_all_processed"], reference_image_processed, vmin=processed_vmin, vmax=processed_vmax, rectangles=[(result.top_roi, ROI_BOX_COLOR, 2.0)]),
        "moving_all_processed": save_linear_grayscale_png(output_dir / naming["moving_all_processed"], moving_image_processed, vmin=processed_vmin, vmax=processed_vmax, rectangles=[(result.bottom_roi, ROI_BOX_COLOR, 2.0)]),
        "reference_roi_processed": save_linear_grayscale_png(output_dir / naming["reference_roi_processed"], result.reference_patch_processed, vmin=processed_vmin, vmax=processed_vmax),
        "moving_roi_processed": save_linear_grayscale_png(output_dir / naming["moving_roi_processed"], result.moving_patch_processed, vmin=processed_vmin, vmax=processed_vmax),
    }


def save_correlation_heatmap_raw(output_dir: Path, result: AlignmentResult, file_name: str) -> Path:
    corr_vmin, corr_vmax = array_display_limits(result.correlation)
    with _noninteractive_save():
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(result.correlation, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        ax.axis("off")
    return _save_figure(fig, output_dir / file_name, dpi=240)


def save_correlation_heatmap_fit(output_dir: Path, result: AlignmentResult, file_name: str) -> Path:
    corr_vmin, corr_vmax = array_display_limits(result.correlation)
    with _noninteractive_save():
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(result.correlation, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        ax.add_patch(patches.Rectangle((result.fit_window[2], result.fit_window[0]), result.fit_window[3] - result.fit_window[2], result.fit_window[1] - result.fit_window[0], fill=False, linewidth=1.8, edgecolor=FIT_WINDOW_COLOR, clip_on=True))
        ax.scatter([result.integer_peak[1]], [result.integer_peak[0]], c=INTEGER_PEAK_COLOR, s=55, marker="x")
        ax.scatter([result.fitted_peak[1]], [result.fitted_peak[0]], c=FIT_COLOR, s=55, marker="+")
        ax.add_patch(patches.Ellipse((result.fitted_peak[1], result.fitted_peak[0]), width=max(result.fit_sigma_axes[0] * 4.0, 2.0), height=max(result.fit_sigma_axes[1] * 4.0, 2.0), angle=result.fit_angle_deg, fill=False, linewidth=2.0, edgecolor=FIT_COLOR, clip_on=True))
        handles = [
            Line2D([0], [0], color=FIT_WINDOW_COLOR, linewidth=1.8, label="fit window"),
            Line2D([0], [0], marker="x", linestyle="None", color=INTEGER_PEAK_COLOR, markersize=8, label="integer peak"),
            Line2D([0], [0], marker="+", linestyle="None", color=FIT_COLOR, markersize=9, label="gaussian center"),
            Line2D([0], [0], color=FIT_COLOR, linewidth=2.0, label="2 sigma contour"),
        ]
        ax.legend(handles=handles, loc="lower right", framealpha=0.85, fontsize=10)
        ax.axis("off")
    return _save_figure(fig, output_dir / file_name, dpi=240)


def save_correlation_fit_window_raw(output_dir: Path, result: AlignmentResult, file_name: str) -> Path:
    y0, y1, x0, x1 = result.fit_window
    crop = result.correlation[y0:y1, x0:x1]
    corr_vmin, corr_vmax = array_display_limits(crop)
    with _noninteractive_save():
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.imshow(crop, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        ax.axis("off")
    return _save_figure(fig, output_dir / file_name, dpi=300)


def save_correlation_fit_window_fit(output_dir: Path, result: AlignmentResult, file_name: str) -> Path:
    y0, y1, x0, x1 = result.fit_window
    crop = result.correlation[y0:y1, x0:x1]
    corr_vmin, corr_vmax = array_display_limits(crop)
    fitted_window = point_to_window(result.fitted_peak, result.fit_window)
    peak_window = point_to_window(result.integer_peak, result.fit_window)
    with _noninteractive_save():
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.imshow(crop, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        if point_inside_window(result.integer_peak, result.fit_window):
            ax.scatter([peak_window[1]], [peak_window[0]], c=INTEGER_PEAK_COLOR, s=60, marker="x")
        ax.scatter([fitted_window[1]], [fitted_window[0]], c=FIT_COLOR, s=60, marker="+")
        for scale in (1.0, 2.0, 3.0):
            if not ellipse_fits_in_window(fitted_window[0], fitted_window[1], result.fit_sigma_axes[0], result.fit_sigma_axes[1], result.fit_angle_deg, crop.shape, scale):
                continue
            ax.add_patch(patches.Ellipse((fitted_window[1], fitted_window[0]), width=max(result.fit_sigma_axes[0] * scale * 2.0, 2.0), height=max(result.fit_sigma_axes[1] * scale * 2.0, 2.0), angle=result.fit_angle_deg, fill=False, linewidth=1.8, edgecolor=FIT_COLOR, clip_on=True))
        handles = [
            Line2D([0], [0], marker="x", linestyle="None", color=INTEGER_PEAK_COLOR, markersize=8, label="integer peak"),
            Line2D([0], [0], marker="+", linestyle="None", color=FIT_COLOR, markersize=9, label="gaussian center"),
            Line2D([0], [0], color=FIT_COLOR, linewidth=2.0, label="sigma contours"),
        ]
        ax.legend(handles=handles, loc="upper left", framealpha=0.85, fontsize=10)
        ax.axis("off")
    return _save_figure(fig, output_dir / file_name, dpi=300)


def save_stitched_preview(output_dir: Path, reference_image: np.ndarray, moving_image: np.ndarray, result: AlignmentResult) -> Path:
    stitched = stitch_pair(reference_image, moving_image, result.offset_y, result.offset_x)
    path = output_dir / "result-1.tif"
    legacy_png = output_dir / "result-1.png"
    if legacy_png.exists():
        legacy_png.unlink()
    tifffile.imwrite(path, np.asarray(stitched, dtype=np.float32), photometric="minisblack")
    return path


def save_stitched_preview_transition(output_dir: Path, reference_image: np.ndarray, moving_image: np.ndarray, result: AlignmentResult, *, blend_axis: str) -> Path:
    stitched = stitch_pair_transition(reference_image, moving_image, result.offset_y, result.offset_x, blend_axis=blend_axis)
    path = output_dir / "result-2.tif"
    legacy_png = output_dir / "result-2.png"
    if legacy_png.exists():
        legacy_png.unlink()
    tifffile.imwrite(path, np.asarray(stitched, dtype=np.float32), photometric="minisblack")
    return path


def save_stitched_preview_cropped(
    output_dir: Path,
    file_name: str,
    stitched: np.ndarray,
    crop_bounds: FitWindow,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    cropped = crop_image(stitched, crop_bounds)
    tif_path = output_dir / file_name.replace(".png", ".tif")
    legacy_png = output_dir / file_name
    if legacy_png.exists():
        legacy_png.unlink()
    tifffile.imwrite(tif_path, np.asarray(cropped, dtype=np.float32), photometric="minisblack")
    return tif_path


def save_stage_para(path: Path, reference_roi: Roi, moving_roi: Roi, fit_window: FitWindow) -> None:
    _json_dump(
        path,
        {
            "version": 1,
            "reference_roi": [int(v) for v in reference_roi],
            "moving_roi": [int(v) for v in moving_roi],
            "fit_window": [int(v) for v in fit_window],
        },
    )


def load_stage_para(path: Path, reference_shape: tuple[int, int], moving_shape: tuple[int, int]) -> tuple[Roi, Roi, FitWindow] | None:
    candidates = [
        path,
        path.with_name("selection.json"),
        path.with_name("para.yaml"),
        path.with_name("paa.yaml"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = _load_mapping_file(candidate)
            reference_roi = parse_roi(payload.get("reference_roi"))
            moving_roi = parse_roi(payload.get("moving_roi"))
            fit_window = parse_roi(payload.get("fit_window"))
            if reference_roi is None or moving_roi is None or fit_window is None:
                continue
            ref_h, ref_w = roi_height_width(reference_roi)
            mov_h, mov_w = roi_height_width(moving_roi)
            reference_roi = clamp_roi(reference_roi[0], reference_roi[2], ref_h, ref_w, reference_shape)
            moving_roi = clamp_roi(moving_roi[0], moving_roi[2], mov_h, mov_w, moving_shape)
            return reference_roi, moving_roi, fit_window
        except Exception:
            continue
    return None


def save_root_para(
    path: Path,
    *,
    pre_stitch_shape: tuple[int, int],
    estimation_frame: tuple[int, int],
    left_result: AlignmentResult,
    right_result: AlignmentResult,
    final_result: AlignmentResult,
    left_crop_bounds: FitWindow | None = None,
    right_crop_bounds: FitWindow | None = None,
    final_crop_bounds: FitWindow | None = None,
) -> None:
    payload = {
        "version": 1,
        "order": "vertical_then_horizontal",
        "pre_stitch_shape": [int(pre_stitch_shape[0]), int(pre_stitch_shape[1])],
        "estimation_frame": [int(estimation_frame[0]), int(estimation_frame[1])],
        "left": {"offset_y": float(left_result.offset_y), "offset_x": float(left_result.offset_x)},
        "right": {"offset_y": float(right_result.offset_y), "offset_x": float(right_result.offset_x)},
        "final": {"offset_y": float(final_result.offset_y), "offset_x": float(final_result.offset_x)},
    }
    if left_crop_bounds is not None:
        payload["left"]["crop_bounds"] = [int(v) for v in left_crop_bounds]
        payload["left"]["cropped_shape"] = [
            int(left_crop_bounds[1] - left_crop_bounds[0]),
            int(left_crop_bounds[3] - left_crop_bounds[2]),
        ]
    if right_crop_bounds is not None:
        payload["right"]["crop_bounds"] = [int(v) for v in right_crop_bounds]
        payload["right"]["cropped_shape"] = [
            int(right_crop_bounds[1] - right_crop_bounds[0]),
            int(right_crop_bounds[3] - right_crop_bounds[2]),
        ]
    if final_crop_bounds is not None:
        payload["final"]["crop_bounds"] = [int(v) for v in final_crop_bounds]
        payload["final"]["cropped_shape"] = [
            int(final_crop_bounds[1] - final_crop_bounds[0]),
            int(final_crop_bounds[3] - final_crop_bounds[2]),
        ]
        payload["crop_bounds"] = [int(v) for v in final_crop_bounds]
        payload["cropped_stitch_shape"] = [
            int(final_crop_bounds[1] - final_crop_bounds[0]),
            int(final_crop_bounds[3] - final_crop_bounds[2]),
        ]
    _json_dump(
        path,
        payload,
    )


def _show_blocking(fig: plt.Figure, accepted_getter) -> None:
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        if accepted_getter():
            break
        plt.pause(0.05)
        time.sleep(0.01)


class VerticalMirrorDualRoiSelector:
    def __init__(self, top_image: np.ndarray, bottom_image: np.ndarray, pair_label: str, preprocessing_mode: str, vessel_keep_percent: float, frangi_sigmas: tuple[float, ...], initial_top_roi: Roi | None = None, initial_bottom_roi: Roi | None = None) -> None:
        self.top_image = np.asarray(top_image)
        self.bottom_image = np.asarray(bottom_image)
        self.pair_label = pair_label
        self.preprocessing_mode = preprocessing_mode
        self.vessel_keep_percent = float(vessel_keep_percent)
        self.frangi_sigmas = tuple(float(v) for v in frangi_sigmas)
        self.top_roi = initial_top_roi
        self.bottom_roi = initial_bottom_roi
        self._drag_mode: str | None = None
        self._drag_axis: str | None = None
        self._draw_anchor: tuple[float, float] | None = None
        self._move_offset: tuple[float, float] | None = None
        self.accepted = False
        self.preview_fig: plt.Figure | None = None

        self.fig, (self.ax_top, self.ax_bottom) = plt.subplots(2, 1, figsize=(12, 12))
        self.fig.subplots_adjust(bottom=0.12, hspace=0.16)
        self.ax_top.imshow(percentile_view(self.top_image), cmap="gray", interpolation="nearest")
        self.ax_bottom.imshow(percentile_view(self.bottom_image), cmap="gray", interpolation="nearest")
        self.ax_top.set_title(f"{self.pair_label}: top image")
        self.ax_bottom.set_title(f"{self.pair_label}: bottom image")
        self.status = self.fig.text(0.02, 0.96, "", fontsize=10, va="top")
        self.top_patch = patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor=ROI_BOX_COLOR, visible=False)
        self.bottom_patch = patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor=ROI_BOX_COLOR, visible=False)
        self.ax_top.add_patch(self.top_patch)
        self.ax_bottom.add_patch(self.bottom_patch)
        preview_button_ax = self.fig.add_axes([0.63, 0.025, 0.16, 0.055])
        self.preview_button = Button(preview_button_ax, "Preview Heatmap")
        self.preview_button.on_clicked(self._on_preview_clicked)
        done_button_ax = self.fig.add_axes([0.82, 0.025, 0.14, 0.055])
        self.done_button = Button(done_button_ax, "Done")
        self.done_button.on_clicked(self._on_done_clicked)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_patches()

    def select(self) -> tuple[Roi, Roi]:
        _show_blocking(self.fig, lambda: self.accepted)
        if not self.accepted or self.top_roi is None or self.bottom_roi is None:
            raise RuntimeError("ROI selection was cancelled before acceptance.")
        return self.top_roi, self.bottom_roi

    def _close_preview(self) -> None:
        if self.preview_fig is not None:
            plt.close(self.preview_fig)
            self.preview_fig = None

    def _set_status(self, text: str) -> None:
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def _refresh_patches(self) -> None:
        for rect, roi in ((self.top_patch, self.top_roi), (self.bottom_patch, self.bottom_roi)):
            if roi is None:
                rect.set_visible(False)
            else:
                rect.set_xy((roi[2], roi[0]))
                rect.set_width(roi[3] - roi[2])
                rect.set_height(roi[1] - roi[0])
                rect.set_visible(True)
        self._set_status(f"{self.pair_label}: preview with 'Preview Heatmap', then click Done." if self.top_roi is not None and self.bottom_roi is not None else f"{self.pair_label}: draw on either image to define the shared ROI.")

    def _preview_alignment(self) -> PreparedAlignment:
        if self.top_roi is None or self.bottom_roi is None:
            raise ValueError("Both ROIs must exist before preview.")
        return prepare_alignment_from_manual_rois(self.top_image, self.bottom_image, self.top_roi, self.bottom_roi, self.preprocessing_mode, self.vessel_keep_percent, self.frangi_sigmas)

    def _show_preview_heatmap(self) -> None:
        prepared = self._preview_alignment()
        self._close_preview()
        with _noninteractive_save():
            preview_fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ref_vmin, ref_vmax = array_display_limits(np.asarray(prepared.reference_info["feature_patch"]))
        mov_vmin, mov_vmax = array_display_limits(np.asarray(prepared.moving_info["feature_patch"]))
        corr_vmin, corr_vmax = array_display_limits(prepared.corr)
        axes[0].imshow(prepared.reference_info["feature_patch"], cmap="gray", interpolation="nearest", vmin=ref_vmin, vmax=ref_vmax)
        axes[1].imshow(prepared.moving_info["feature_patch"], cmap="gray", interpolation="nearest", vmin=mov_vmin, vmax=mov_vmax)
        axes[2].imshow(prepared.corr, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        for ax in axes:
            ax.axis("off")
        self.preview_fig = preview_fig
        preview_fig.show()

    def _on_preview_clicked(self, _event) -> None:
        try:
            self._show_preview_heatmap()
            self._set_status(f"{self.pair_label}: preview updated.")
        except Exception as exc:
            self._set_status(f"{self.pair_label}: preview failed: {exc}")

    def _on_done_clicked(self, _event) -> None:
        if self.top_roi is None or self.bottom_roi is None:
            self._set_status(f"{self.pair_label}: both ROIs must exist before Done.")
            return
        self.accepted = True
        self._close_preview()
        plt.close(self.fig)

    def _on_key(self, event) -> None:
        if event.key == "enter":
            self._on_done_clicked(event)
        elif event.key == "p":
            self._on_preview_clicked(event)
        elif event.key == "escape":
            self._close_preview()
            plt.close(self.fig)

    def _on_press(self, event) -> None:
        axis_name = "top" if event.inaxes is self.ax_top else "bottom" if event.inaxes is self.ax_bottom else None
        if event.button != 1 or axis_name is None or event.xdata is None or event.ydata is None:
            return
        current_roi = self.top_roi if axis_name == "top" else self.bottom_roi
        if point_in_roi(float(event.xdata), float(event.ydata), current_roi):
            self._drag_mode = "move"
            self._drag_axis = axis_name
            assert current_roi is not None
            self._move_offset = (float(event.xdata) - current_roi[2], float(event.ydata) - current_roi[0])
        else:
            self._drag_mode = "draw"
            self._drag_axis = axis_name
            self._draw_anchor = (float(event.xdata), float(event.ydata))

    def _on_motion(self, event) -> None:
        if self._drag_mode is None or self._drag_axis is None:
            return
        axis_name = "top" if event.inaxes is self.ax_top else "bottom" if event.inaxes is self.ax_bottom else None
        if axis_name != self._drag_axis or event.xdata is None or event.ydata is None:
            return
        image = self.top_image if axis_name == "top" else self.bottom_image
        if self._drag_mode == "draw":
            assert self._draw_anchor is not None
            x0 = min(self._draw_anchor[0], float(event.xdata))
            x1 = max(self._draw_anchor[0], float(event.xdata))
            y0 = min(self._draw_anchor[1], float(event.ydata))
            y1 = max(self._draw_anchor[1], float(event.ydata))
            roi = clamp_roi(y0, x0, y1 - y0, x1 - x0, image.shape)
        else:
            assert self._move_offset is not None
            roi = self.top_roi if axis_name == "top" else self.bottom_roi
            if roi is None:
                return
            height, width = roi_height_width(roi)
            roi = clamp_roi(float(event.ydata) - self._move_offset[1], float(event.xdata) - self._move_offset[0], height, width, image.shape)
        if axis_name == "top":
            self.top_roi = roi
            self.bottom_roi = mirror_roi_vertical_symmetric(roi, self.top_image.shape, self.bottom_image.shape)
        else:
            self.bottom_roi = roi
            self.top_roi = mirror_roi_vertical_symmetric(roi, self.bottom_image.shape, self.top_image.shape)
        self._refresh_patches()

    def _on_release(self, _event) -> None:
        self._drag_mode = None
        self._drag_axis = None
        self._draw_anchor = None
        self._move_offset = None


class HorizontalMirrorDualRoiSelector(VerticalMirrorDualRoiSelector):
    def __init__(self, left_image: np.ndarray, right_image: np.ndarray, pair_label: str, preprocessing_mode: str, vessel_keep_percent: float, frangi_sigmas: tuple[float, ...], initial_left_roi: Roi | None = None, initial_right_roi: Roi | None = None) -> None:
        self.top_image = np.asarray(left_image)
        self.bottom_image = np.asarray(right_image)
        self.pair_label = pair_label
        self.preprocessing_mode = preprocessing_mode
        self.vessel_keep_percent = float(vessel_keep_percent)
        self.frangi_sigmas = tuple(float(v) for v in frangi_sigmas)
        self.top_roi = initial_left_roi
        self.bottom_roi = initial_right_roi
        self._drag_mode = None
        self._drag_axis = None
        self._draw_anchor = None
        self._move_offset = None
        self.accepted = False
        self.preview_fig = None
        self.fig, (self.ax_top, self.ax_bottom) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.subplots_adjust(bottom=0.12, wspace=0.08)
        self.ax_top.imshow(percentile_view(self.top_image), cmap="gray", interpolation="nearest")
        self.ax_bottom.imshow(percentile_view(self.bottom_image), cmap="gray", interpolation="nearest")
        self.ax_top.set_title(f"{self.pair_label}: left image")
        self.ax_bottom.set_title(f"{self.pair_label}: right image")
        self.status = self.fig.text(0.02, 0.96, "", fontsize=10, va="top")
        self.top_patch = patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor=ROI_BOX_COLOR, visible=False)
        self.bottom_patch = patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor=ROI_BOX_COLOR, visible=False)
        self.ax_top.add_patch(self.top_patch)
        self.ax_bottom.add_patch(self.bottom_patch)
        preview_button_ax = self.fig.add_axes([0.63, 0.025, 0.16, 0.055])
        self.preview_button = Button(preview_button_ax, "Preview Heatmap")
        self.preview_button.on_clicked(self._on_preview_clicked)
        done_button_ax = self.fig.add_axes([0.82, 0.025, 0.14, 0.055])
        self.done_button = Button(done_button_ax, "Done")
        self.done_button.on_clicked(self._on_done_clicked)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_patches()

    def _show_preview_heatmap(self) -> None:
        prepared = self._preview_alignment()
        self._close_preview()
        with _noninteractive_save():
            preview_fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ref_vmin, ref_vmax = array_display_limits(np.asarray(prepared.reference_info["feature_patch"]))
        mov_vmin, mov_vmax = array_display_limits(np.asarray(prepared.moving_info["feature_patch"]))
        corr_vmin, corr_vmax = array_display_limits(prepared.corr)
        axes[0].imshow(prepared.reference_info["feature_patch"], cmap="gray", interpolation="nearest", vmin=ref_vmin, vmax=ref_vmax)
        axes[1].imshow(prepared.moving_info["feature_patch"], cmap="gray", interpolation="nearest", vmin=mov_vmin, vmax=mov_vmax)
        axes[2].imshow(prepared.corr, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        for ax in axes:
            ax.axis("off")
        self.preview_fig = preview_fig
        preview_fig.show()

    def _on_motion(self, event) -> None:
        if self._drag_mode is None or self._drag_axis is None:
            return
        axis_name = "top" if event.inaxes is self.ax_top else "bottom" if event.inaxes is self.ax_bottom else None
        if axis_name != self._drag_axis or event.xdata is None or event.ydata is None:
            return
        image = self.top_image if axis_name == "top" else self.bottom_image
        if self._drag_mode == "draw":
            assert self._draw_anchor is not None
            x0 = min(self._draw_anchor[0], float(event.xdata))
            x1 = max(self._draw_anchor[0], float(event.xdata))
            y0 = min(self._draw_anchor[1], float(event.ydata))
            y1 = max(self._draw_anchor[1], float(event.ydata))
            roi = clamp_roi(y0, x0, y1 - y0, x1 - x0, image.shape)
        else:
            assert self._move_offset is not None
            roi = self.top_roi if axis_name == "top" else self.bottom_roi
            if roi is None:
                return
            height, width = roi_height_width(roi)
            roi = clamp_roi(float(event.ydata) - self._move_offset[1], float(event.xdata) - self._move_offset[0], height, width, image.shape)
        if axis_name == "top":
            self.top_roi = roi
            self.bottom_roi = mirror_roi_horizontal_symmetric(roi, self.top_image.shape, self.bottom_image.shape)
        else:
            self.bottom_roi = roi
            self.top_roi = mirror_roi_horizontal_symmetric(roi, self.bottom_image.shape, self.top_image.shape)
        self._refresh_patches()


class CorrelationFitWindowSelector:
    def __init__(self, corr: np.ndarray, pair_label: str, initial_window: FitWindow | None = None) -> None:
        self.corr = np.asarray(corr, dtype=np.float32)
        self.pair_label = pair_label
        self.fit_window = initial_window
        self._drag_mode: str | None = None
        self._draw_anchor: tuple[float, float] | None = None
        self._move_offset: tuple[float, float] | None = None
        self.accepted = False
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.subplots_adjust(bottom=0.14)
        corr_vmin, corr_vmax = array_display_limits(self.corr)
        self.ax.imshow(self.corr, cmap="gray", interpolation="nearest", vmin=corr_vmin, vmax=corr_vmax)
        self.ax.axis("off")
        self.window_patch = patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=2.0, edgecolor=FIT_WINDOW_COLOR, visible=False)
        self.ax.add_patch(self.window_patch)
        self.status = self.fig.text(0.02, 0.96, "", fontsize=10, va="top")
        done_button_ax = self.fig.add_axes([0.83, 0.03, 0.13, 0.055])
        self.done_button = Button(done_button_ax, "Done")
        self.done_button.on_clicked(self._on_done_clicked)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_patch()

    def select(self) -> FitWindow:
        _show_blocking(self.fig, lambda: self.accepted)
        if not self.accepted or self.fit_window is None:
            raise RuntimeError("Correlation fit-window selection was cancelled before acceptance.")
        return self.fit_window

    def _refresh_patch(self) -> None:
        if self.fit_window is None:
            self.window_patch.set_visible(False)
            self.status.set_text("No fit window selected yet.")
        else:
            y0, y1, x0, x1 = self.fit_window
            self.window_patch.set_xy((x0, y0))
            self.window_patch.set_width(x1 - x0)
            self.window_patch.set_height(y1 - y0)
            self.window_patch.set_visible(True)
            self.status.set_text(f"Selected fit window: ({y0}, {y1}, {x0}, {x1})")
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if point_in_roi(float(event.xdata), float(event.ydata), self.fit_window):
            self._drag_mode = "move"
            assert self.fit_window is not None
            self._move_offset = (float(event.xdata) - self.fit_window[2], float(event.ydata) - self.fit_window[0])
        else:
            self._drag_mode = "draw"
            self._draw_anchor = (float(event.xdata), float(event.ydata))

    def _on_motion(self, event) -> None:
        if self._drag_mode is None or event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self._drag_mode == "draw":
            assert self._draw_anchor is not None
            x0 = min(self._draw_anchor[0], float(event.xdata))
            x1 = max(self._draw_anchor[0], float(event.xdata))
            y0 = min(self._draw_anchor[1], float(event.ydata))
            y1 = max(self._draw_anchor[1], float(event.ydata))
            self.fit_window = clamp_roi(y0, x0, y1 - y0, x1 - x0, self.corr.shape, min_size=4)
        else:
            assert self._move_offset is not None
            if self.fit_window is None:
                return
            height, width = roi_height_width(self.fit_window)
            self.fit_window = clamp_roi(float(event.ydata) - self._move_offset[1], float(event.xdata) - self._move_offset[0], height, width, self.corr.shape, min_size=4)
        self._refresh_patch()

    def _on_done_clicked(self, _event) -> None:
        if self.fit_window is None:
            self.status.set_text("Select one fit window before Done.")
            self.fig.canvas.draw_idle()
            return
        self.accepted = True
        plt.close(self.fig)

    def _on_key(self, event) -> None:
        if event.key == "enter":
            self._on_done_clicked(event)
        elif event.key == "escape":
            plt.close(self.fig)

    def _on_release(self, _event) -> None:
        self._drag_mode = None
        self._draw_anchor = None
        self._move_offset = None


def run_pair_alignment(
    *,
    pair_label: str,
    reference_label: str,
    moving_label: str,
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    initial_reference_roi: Roi | None,
    initial_moving_roi: Roi | None,
    initial_fit_window: FitWindow | None,
    initial_fit_point: FitPoint | None,
    output_dir: Path,
    preprocessing_mode: str,
    vessel_keep_percent: float,
    frangi_sigmas: tuple[float, ...],
    fit_point_radius: int,
    stitch_axis: str,
    roi_selector_mode: str,
    stage_para_path: Path,
    allow_para_reuse: bool,
) -> tuple[AlignmentResult, dict[str, Path], bool]:
    naming = build_output_naming(reference_label, moving_label, preprocessing_mode)
    reused_existing_para = False
    reference_roi = initial_reference_roi
    moving_roi = initial_moving_roi
    selected_fit_window = initial_fit_window

    if allow_para_reuse:
        loaded = load_stage_para(stage_para_path, reference_image.shape, moving_image.shape)
        if loaded is not None:
            reference_roi, moving_roi, selected_fit_window = loaded
            reused_existing_para = True

    if roi_selector_mode == "vertical_mirror":
        reference_roi, moving_roi = resolve_vertical_mirror_rois(reference_roi, moving_roi, reference_image.shape, moving_image.shape)
    elif roi_selector_mode == "horizontal_mirror":
        reference_roi, moving_roi = resolve_horizontal_mirror_rois(reference_roi, moving_roi, reference_image.shape, moving_image.shape)
    else:
        raise ValueError(f"Unsupported roi_selector_mode: {roi_selector_mode}")

    if reference_roi is None or moving_roi is None:
        if roi_selector_mode == "vertical_mirror":
            selector = VerticalMirrorDualRoiSelector(reference_image, moving_image, pair_label, preprocessing_mode, vessel_keep_percent, frangi_sigmas, reference_roi, moving_roi)
        else:
            selector = HorizontalMirrorDualRoiSelector(reference_image, moving_image, pair_label, preprocessing_mode, vessel_keep_percent, frangi_sigmas, reference_roi, moving_roi)
        reference_roi, moving_roi = selector.select()
        reused_existing_para = False

    if roi_selector_mode == "vertical_mirror":
        reference_roi, moving_roi = resolve_vertical_mirror_rois(reference_roi, moving_roi, reference_image.shape, moving_image.shape)
    else:
        reference_roi, moving_roi = resolve_horizontal_mirror_rois(reference_roi, moving_roi, reference_image.shape, moving_image.shape)

    prepared = prepare_alignment_from_manual_rois(reference_image, moving_image, reference_roi, moving_roi, preprocessing_mode, vessel_keep_percent, frangi_sigmas)
    if selected_fit_window is None and initial_fit_point is not None and not reused_existing_para:
        selected_fit_window = point_fit_window(initial_fit_point, prepared.corr.shape, radius=fit_point_radius)
    if selected_fit_window is None:
        selected_fit_window = CorrelationFitWindowSelector(prepared.corr, pair_label).select()
        reused_existing_para = False
    selected_fit_window = validate_fit_window(selected_fit_window, prepared.corr.shape)

    result = finalize_alignment_with_fit_window(prepared, selected_fit_window, preprocessing_mode, reference_image.shape, stitch_axis)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_stage_para(stage_para_path, reference_roi, moving_roi, result.fit_window)

    output_paths: dict[str, Path] = {}
    output_paths.update(save_pair_roi_overview(output_dir, reference_image, moving_image, result, naming=naming))
    output_paths.update(save_pair_processed_roi_overview(output_dir, reference_image, moving_image, result, vessel_keep_percent, frangi_sigmas, naming=naming))
    output_paths["heatmap_all_raw"] = save_correlation_heatmap_raw(output_dir, result, naming["heatmap_all_raw"])
    output_paths["heatmap_all_label"] = save_correlation_heatmap_fit(output_dir, result, naming["heatmap_all_label"])
    output_paths["heatmap_roi_raw"] = save_correlation_fit_window_raw(output_dir, result, naming["heatmap_roi_raw"])
    output_paths["heatmap_roi_label"] = save_correlation_fit_window_fit(output_dir, result, naming["heatmap_roi_label"])
    output_paths["result_1"] = save_stitched_preview(output_dir, reference_image, moving_image, result)
    output_paths["result_2"] = save_stitched_preview_transition(output_dir, reference_image, moving_image, result, blend_axis=stitch_axis)
    return result, output_paths, reused_existing_para


def _normalize_mode_triplet(value: Any) -> list[str]:
    if value is None:
        return ["frangi", "frangi", "raw"]
    if isinstance(value, str):
        return ["raw", "raw", "raw"] if value == "raw" else [value, value, "raw"]
    items = [str(v) for v in value]
    if not items:
        return ["frangi", "frangi", "raw"]
    if len(items) == 1:
        return _normalize_mode_triplet(items[0])
    if len(items) == 2:
        return [items[0], items[1], "raw"]
    if len(items) == 3:
        return items
    raise ValueError("mode must be a string or a 3-item sequence ordered as [L, R, LR].")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive manual ROI phase-correlation stitch tool.")
    parser.add_argument("--x-load-path", type=Path, required=True)
    parser.add_argument("--y-save-path", type=Path, required=True)
    parser.add_argument("--stitch-save-fold", type=Path, required=True)
    parser.add_argument("--frame", type=int, nargs=2, default=(0, -1), metavar=("START", "END"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--mode", nargs="+", choices=("raw", "frangi"), default=list(_normalize_mode_triplet(None)))
    parser.add_argument("--vessel-keep-percent", type=float, default=DEFAULT_VESSEL_KEEP_PERCENT)
    parser.add_argument("--frangi-sigmas", type=float, nargs="+", default=DEFAULT_FRANGI_SIGMAS)
    parser.add_argument("--fit-point-radius", type=int, default=DEFAULT_FIT_POINT_RADIUS)
    parser.add_argument("--left-top-roi", type=int, nargs=4, default=None)
    parser.add_argument("--left-bottom-roi", type=int, nargs=4, default=None)
    parser.add_argument("--left-fit-window", type=int, nargs=4, default=None)
    parser.add_argument("--left-fit-point", type=int, nargs=2, default=None)
    parser.add_argument("--right-top-roi", type=int, nargs=4, default=None)
    parser.add_argument("--right-bottom-roi", type=int, nargs=4, default=None)
    parser.add_argument("--right-fit-window", type=int, nargs=4, default=None)
    parser.add_argument("--right-fit-point", type=int, nargs=2, default=None)
    parser.add_argument("--final-left-roi", type=int, nargs=4, default=None)
    parser.add_argument("--final-right-roi", type=int, nargs=4, default=None)
    parser.add_argument("--final-fit-window", type=int, nargs=4, default=None)
    parser.add_argument("--final-fit-point", type=int, nargs=2, default=None)
    return parser


def write_stitched_tiff(
    x_load_path: Path,
    y_save_path: Path,
    left_result: AlignmentResult,
    right_result: AlignmentResult,
    final_result: AlignmentResult,
    *,
    chunk_size: int,
    left_crop_bounds: FitWindow | None = None,
    right_crop_bounds: FitWindow | None = None,
    final_crop_bounds: FitWindow | None = None,
) -> tuple[int, tuple[int, ...], np.dtype]:
    with tifffile.TiffFile(x_load_path) as tif:
        data_shape = tif.series[0].shape
        if len(data_shape) == 2:
            frame = tif.asarray().astype(np.float32, copy=False)
            stitched = stitch_frame_from_offsets(
                frame,
                left_offset_y=left_result.offset_y,
                left_offset_x=left_result.offset_x,
                right_offset_y=right_result.offset_y,
                right_offset_x=right_result.offset_x,
                final_offset_y=final_result.offset_y,
                final_offset_x=final_result.offset_x,
                left_crop_bounds=left_crop_bounds,
                right_crop_bounds=right_crop_bounds,
                final_crop_bounds=final_crop_bounds,
            )
            with tifffile.TiffWriter(y_save_path, bigtiff=True) as writer:
                writer.write(stitched.astype(np.float32, copy=False), photometric="minisblack")
            return 1, stitched.shape, stitched.dtype
        if len(data_shape) != 3:
            raise ValueError("Input TIFF must be 2D or 3D.")

        total_frames = int(data_shape[0])
        written = 0
        stitched_shape: tuple[int, ...] | None = None
        stitched_dtype: np.dtype | None = None
        with tifffile.TiffWriter(y_save_path, bigtiff=True) as writer:
            with tqdm(total=total_frames, desc="stitch", unit="frame") as progress:
                for chunk_start in range(0, total_frames, chunk_size):
                    chunk_stop = min(chunk_start + chunk_size, total_frames)
                    chunk = tif.asarray(key=range(chunk_start, chunk_stop))
                    if chunk.ndim == 2:
                        chunk = chunk[None, ...]
                    for i in range(chunk.shape[0]):
                        stitched = stitch_frame_from_offsets(
                            chunk[i],
                            left_offset_y=left_result.offset_y,
                            left_offset_x=left_result.offset_x,
                            right_offset_y=right_result.offset_y,
                            right_offset_x=right_result.offset_x,
                            final_offset_y=final_result.offset_y,
                            final_offset_x=final_result.offset_x,
                            left_crop_bounds=left_crop_bounds,
                            right_crop_bounds=right_crop_bounds,
                            final_crop_bounds=final_crop_bounds,
                        )
                        writer.write(stitched.astype(np.float32, copy=False), photometric="minisblack", contiguous=True)
                        written += 1
                        stitched_shape = stitched.shape
                        stitched_dtype = stitched.dtype
                        progress.update(1)
                    del chunk
                    gc.collect()
        if stitched_shape is None or stitched_dtype is None:
            raise RuntimeError("No frames were written to the stitched TIFF.")
        return written, stitched_shape, stitched_dtype


def run(args: argparse.Namespace) -> int:
    x_load_path = resolved_path(args.x_load_path)
    y_save_path = resolved_path(args.y_save_path)
    stitch_save_fold = resolved_path(args.stitch_save_fold)
    if x_load_path == y_save_path:
        raise ValueError("x_load_path and y_save_path must be different.")

    cache_dir = stitch_save_fold / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = int(max(args.chunk_size, 1))
    frame_cfg = tuple(int(v) for v in args.frame)
    mode_triplet = _normalize_mode_triplet(args.mode)
    vessel_keep_percent = float(args.vessel_keep_percent)
    frangi_sigmas = tuple(float(v) for v in args.frangi_sigmas)
    fit_point_radius = int(max(args.fit_point_radius, 1))

    projection, estimation_frame = load_or_project_sum(x_load_path, frame_cfg, chunk_size, cache_dir, use_cache=True)
    quadrants = split_rotate_quadrants(projection)
    stitch_save_fold.mkdir(parents=True, exist_ok=True)
    y_save_path.parent.mkdir(parents=True, exist_ok=True)

    left_top_roi = parse_roi(args.left_top_roi)
    left_bottom_roi = parse_roi(args.left_bottom_roi)
    left_fit_window = parse_roi(args.left_fit_window)
    left_fit_point = parse_fit_point(args.left_fit_point)
    right_top_roi = parse_roi(args.right_top_roi)
    right_bottom_roi = parse_roi(args.right_bottom_roi)
    right_fit_window = parse_roi(args.right_fit_window)
    right_fit_point = parse_fit_point(args.right_fit_point)
    final_left_roi = parse_roi(args.final_left_roi)
    final_right_roi = parse_roi(args.final_right_roi)
    final_fit_window = parse_roi(args.final_fit_window)
    final_fit_point = parse_fit_point(args.final_fit_point)

    left_para_path = stitch_save_fold / "L" / "para.json"
    right_para_path = stitch_save_fold / "R" / "para.json"
    final_para_path = stitch_save_fold / "LR" / "para.json"

    left_result, _left_paths, _left_reused = run_pair_alignment(pair_label="Left column", reference_label="top", moving_label="bottom", reference_image=quadrants["upper_left"], moving_image=quadrants["lower_left"], initial_reference_roi=left_top_roi, initial_moving_roi=left_bottom_roi, initial_fit_window=left_fit_window, initial_fit_point=left_fit_point, output_dir=stitch_save_fold / "L", preprocessing_mode=mode_triplet[0], vessel_keep_percent=vessel_keep_percent, frangi_sigmas=frangi_sigmas, fit_point_radius=fit_point_radius, stitch_axis="vertical", roi_selector_mode="vertical_mirror", stage_para_path=left_para_path, allow_para_reuse=True)
    right_result, _right_paths, _right_reused = run_pair_alignment(pair_label="Right column", reference_label="top", moving_label="bottom", reference_image=quadrants["upper_right"], moving_image=quadrants["lower_right"], initial_reference_roi=right_top_roi, initial_moving_roi=right_bottom_roi, initial_fit_window=right_fit_window, initial_fit_point=right_fit_point, output_dir=stitch_save_fold / "R", preprocessing_mode=mode_triplet[1], vessel_keep_percent=vessel_keep_percent, frangi_sigmas=frangi_sigmas, fit_point_radius=fit_point_radius, stitch_axis="vertical", roi_selector_mode="vertical_mirror", stage_para_path=right_para_path, allow_para_reuse=True)

    left_stitched_raw = stitch_pair(quadrants["upper_left"], quadrants["lower_left"], left_result.offset_y, left_result.offset_x)
    left_stitched_transition = stitch_pair_transition(quadrants["upper_left"], quadrants["lower_left"], left_result.offset_y, left_result.offset_x, blend_axis="vertical")
    left_crop_bounds = compute_pair_crop_bounds(quadrants["upper_left"].shape, quadrants["lower_left"].shape, left_result.offset_y, left_result.offset_x, blend_axis="vertical")
    save_stitched_preview_cropped(stitch_save_fold / "L", "result-3.png", left_stitched_raw, left_crop_bounds)
    save_stitched_preview_cropped(stitch_save_fold / "L", "result-4.png", left_stitched_transition, left_crop_bounds)
    left_stitched_transition_cropped = crop_image(left_stitched_transition, left_crop_bounds)

    right_stitched_raw = stitch_pair(quadrants["upper_right"], quadrants["lower_right"], right_result.offset_y, right_result.offset_x)
    right_stitched_transition = stitch_pair_transition(quadrants["upper_right"], quadrants["lower_right"], right_result.offset_y, right_result.offset_x, blend_axis="vertical")
    right_crop_bounds = compute_pair_crop_bounds(quadrants["upper_right"].shape, quadrants["lower_right"].shape, right_result.offset_y, right_result.offset_x, blend_axis="vertical")
    save_stitched_preview_cropped(stitch_save_fold / "R", "result-3.png", right_stitched_raw, right_crop_bounds)
    save_stitched_preview_cropped(stitch_save_fold / "R", "result-4.png", right_stitched_transition, right_crop_bounds)
    right_stitched_transition_cropped = crop_image(right_stitched_transition, right_crop_bounds)

    root_para_path = stitch_save_fold / "para.json"
    final_result, _final_paths, _final_reused = run_pair_alignment(pair_label="Final left-right", reference_label="left", moving_label="right", reference_image=left_stitched_transition_cropped, moving_image=right_stitched_transition_cropped, initial_reference_roi=final_left_roi, initial_moving_roi=final_right_roi, initial_fit_window=final_fit_window, initial_fit_point=final_fit_point, output_dir=stitch_save_fold / "LR", preprocessing_mode=mode_triplet[2], vessel_keep_percent=vessel_keep_percent, frangi_sigmas=frangi_sigmas, fit_point_radius=fit_point_radius, stitch_axis="horizontal", roi_selector_mode="horizontal_mirror", stage_para_path=final_para_path, allow_para_reuse=True)

    final_stitched_raw = stitch_pair(left_stitched_transition_cropped, right_stitched_transition_cropped, final_result.offset_y, final_result.offset_x)
    final_stitched_transition = stitch_pair_transition(left_stitched_transition_cropped, right_stitched_transition_cropped, final_result.offset_y, final_result.offset_x, blend_axis="horizontal")
    final_crop_bounds = compute_pair_crop_bounds(left_stitched_transition_cropped.shape, right_stitched_transition_cropped.shape, final_result.offset_y, final_result.offset_x, blend_axis="horizontal")
    save_stitched_preview_cropped(stitch_save_fold / "LR", "result-3.png", final_stitched_raw, final_crop_bounds)
    save_stitched_preview_cropped(stitch_save_fold / "LR", "result-4.png", final_stitched_transition, final_crop_bounds)

    save_root_para(root_para_path, pre_stitch_shape=projection.shape, estimation_frame=estimation_frame, left_result=left_result, right_result=right_result, final_result=final_result, left_crop_bounds=left_crop_bounds, right_crop_bounds=right_crop_bounds, final_crop_bounds=final_crop_bounds)
    write_stitched_tiff(x_load_path, y_save_path, left_result, right_result, final_result, chunk_size=chunk_size, left_crop_bounds=left_crop_bounds, right_crop_bounds=right_crop_bounds, final_crop_bounds=final_crop_bounds)
    return 0

def _as_list4(value: Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v) for v in value]


def _as_list2(value: Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v) for v in value]


def _as_frame_range(value: Sequence[int] | None) -> list[int]:
    if value is None:
        return [0, -1]
    result = [int(v) for v in value]
    if len(result) != 2:
        raise ValueError("frame must be [start, end), with end < 1 meaning full range.")
    return result


class Stitch:
    def __init__(
        self,
        x_load_path: str | Path,
        y_save_path: str | Path,
        stitch_save_fold: str | Path,
        frame: Sequence[int] = (0, -1),
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mode: str | Sequence[str] = ("frangi", "frangi", "raw"),
        vessel_keep_percent: float = DEFAULT_VESSEL_KEEP_PERCENT,
        frangi_sigmas: Sequence[float] = DEFAULT_FRANGI_SIGMAS,
        fit_point_radius: int = DEFAULT_FIT_POINT_RADIUS,
        left_top_roi: Sequence[int] | None = None,
        left_bottom_roi: Sequence[int] | None = None,
        left_fit_window: Sequence[int] | None = None,
        left_fit_point: Sequence[int] | None = None,
        right_top_roi: Sequence[int] | None = None,
        right_bottom_roi: Sequence[int] | None = None,
        right_fit_window: Sequence[int] | None = None,
        right_fit_point: Sequence[int] | None = None,
        final_left_roi: Sequence[int] | None = None,
        final_right_roi: Sequence[int] | None = None,
        final_fit_window: Sequence[int] | None = None,
        final_fit_point: Sequence[int] | None = None,
        enable: bool = True,
    ) -> None:
        self.enable = bool(enable)
        self.x_load_path = Path(x_load_path)
        self.y_save_path = Path(y_save_path)
        self.stitch_save_fold = Path(stitch_save_fold)
        self.frame = _as_frame_range(frame)
        self.chunk_size = int(chunk_size)
        if isinstance(mode, str):
            self.mode: list[str] = [mode]
        else:
            self.mode = [str(v) for v in mode]
        self.vessel_keep_percent = float(vessel_keep_percent)
        self.frangi_sigmas = tuple(float(v) for v in frangi_sigmas)
        self.fit_point_radius = int(fit_point_radius)
        self.left_top_roi = _as_list4(left_top_roi)
        self.left_bottom_roi = _as_list4(left_bottom_roi)
        self.left_fit_window = _as_list4(left_fit_window)
        self.left_fit_point = _as_list2(left_fit_point)
        self.right_top_roi = _as_list4(right_top_roi)
        self.right_bottom_roi = _as_list4(right_bottom_roi)
        self.right_fit_window = _as_list4(right_fit_window)
        self.right_fit_point = _as_list2(right_fit_point)
        self.final_left_roi = _as_list4(final_left_roi)
        self.final_right_roi = _as_list4(final_right_roi)
        self.final_fit_window = _as_list4(final_fit_window)
        self.final_fit_point = _as_list2(final_fit_point)

    def forward(self) -> None:
        if not self.enable:
            return
        args = argparse.Namespace(
            x_load_path=self.x_load_path,
            y_save_path=self.y_save_path,
            stitch_save_fold=self.stitch_save_fold,
            frame=self.frame,
            chunk_size=self.chunk_size,
            mode=self.mode,
            vessel_keep_percent=self.vessel_keep_percent,
            frangi_sigmas=list(self.frangi_sigmas),
            fit_point_radius=self.fit_point_radius,
            left_top_roi=self.left_top_roi,
            left_bottom_roi=self.left_bottom_roi,
            left_fit_window=self.left_fit_window,
            left_fit_point=self.left_fit_point,
            right_top_roi=self.right_top_roi,
            right_bottom_roi=self.right_bottom_roi,
            right_fit_window=self.right_fit_window,
            right_fit_point=self.right_fit_point,
            final_left_roi=self.final_left_roi,
            final_right_roi=self.final_right_roi,
            final_fit_window=self.final_fit_window,
            final_fit_point=self.final_fit_point,
        )
        run(args)


def main() -> int:
    parser = build_argument_parser()
    return run(parser.parse_args())


__all__ = [
    "MatchPointStitch",
    "Stitch",
    "split_rotate_quadrants",
    "stitch_frame_from_offsets",
    "stitch_pair",
]


if __name__ == "__main__":
    raise SystemExit(main())
