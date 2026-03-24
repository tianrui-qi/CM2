from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile
import tqdm
import zarr
from skimage import morphology
from skimage.filters import frangi


def robust_limits(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.7,
) -> tuple[float, float]:
    lo, hi = np.percentile(image, [lower, upper])
    if not np.isfinite(lo):
        lo = float(np.nanmin(image))
    if not np.isfinite(hi):
        hi = float(np.nanmax(image))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def rescale_to_unit(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.7,
) -> np.ndarray:
    lo, hi = robust_limits(image, lower, upper)
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _open_video_as_3d(path: str) -> tuple[tifffile.TiffFile, zarr.Array, bool]:
    tif = tifffile.TiffFile(path)
    video = zarr.open(tif.aszarr(), mode="r")
    is_2d = video.ndim == 2
    if is_2d:
        video = video[None, ...]
    if video.ndim != 3:
        tif.close()
        raise ValueError(f"Expected 2D or 3D TIFF, got ndim={video.ndim}: {path}")
    return tif, video, is_2d


def compute_summary_mean(
    y_load_path: str,
    chunk_size: int,
) -> np.ndarray:
    tif, video, is_2d = _open_video_as_3d(y_load_path)
    try:
        if is_2d:
            return np.asarray(video[0], dtype=np.float32)

        total_chunks = (video.shape[0] + chunk_size - 1) // chunk_size
        accum = np.zeros(video.shape[1:], dtype=np.float64)
        with tqdm.tqdm(
            total=total_chunks,
            unit="chunk",
            desc="vessel",
            dynamic_ncols=True,
        ) as bar:
            for start in range(0, video.shape[0], chunk_size):
                end = min(start + chunk_size, video.shape[0])
                chunk = np.asarray(video[start:end], dtype=np.float32)
                if chunk.ndim == 2:
                    chunk = chunk[None, ...]
                accum += np.sum(chunk, axis=0, dtype=np.float64)
                del chunk
                bar.update(1)
        return (accum / float(video.shape[0])).astype(np.float32)
    finally:
        tif.close()


def build_vessel_outputs(
    summary_raw: np.ndarray,
    frangi_sigmas: Sequence[float],
    frangi_black_ridges: bool,
    mask_percentile: float,
    min_object_area: int,
    mask_close_radius: int,
    mask_dilation_radius: int,
    input_percentiles: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_frangi_input = rescale_to_unit(
        summary_raw,
        float(input_percentiles[0]),
        float(input_percentiles[1]),
    )
    vesselness = frangi(
        frame_frangi_input,
        sigmas=tuple(float(x) for x in frangi_sigmas),
        black_ridges=bool(frangi_black_ridges),
    )
    vesselness = np.nan_to_num(
        vesselness,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    positive_vesselness = vesselness[vesselness > 0]
    mask_threshold = (
        float(np.percentile(positive_vesselness, mask_percentile))
        if positive_vesselness.size
        else 0.0
    )
    mask_initial = vesselness >= mask_threshold

    mask_clean = mask_initial.copy()
    if int(min_object_area) > 0:
        mask_clean = (
            morphology.area_opening(
                mask_clean.astype(np.uint8),
                area_threshold=int(min_object_area),
            )
            > 0
        )
    if int(mask_close_radius) > 0:
        mask_clean = morphology.closing(
            mask_clean,
            footprint=morphology.disk(int(mask_close_radius)),
        )
    if int(mask_dilation_radius) > 0:
        mask_clean = morphology.dilation(
            mask_clean,
            footprint=morphology.disk(int(mask_dilation_radius)),
        )

    return (
        vesselness,
        mask_initial.astype(bool),
        mask_clean.astype(bool),
    )


def save_result_images(
    result_save_fold: str,
    vesselness: np.ndarray,
    mask_initial: np.ndarray,
    mask_clean: np.ndarray,
) -> None:
    save_dir = Path(result_save_fold)
    save_dir.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        save_dir / "frangi_vesselness.tif",
        np.asarray(vesselness, dtype=np.float32),
        photometric="minisblack",
    )
    tifffile.imwrite(
        save_dir / "initial_threshold.tif",
        (np.asarray(mask_initial, dtype=np.uint8) * 255),
        photometric="minisblack",
    )
    tifffile.imwrite(
        save_dir / "cleaned_mask.tif",
        (np.asarray(mask_clean, dtype=np.uint8) * 255),
        photometric="minisblack",
    )


def run(
    y_load_path: str,
    result_save_fold: str,
    chunk_size: int,
    frangi_sigmas: Sequence[float],
    frangi_black_ridges: bool,
    mask_percentile: float,
    min_object_area: int,
    mask_close_radius: int,
    mask_dilation_radius: int,
    input_percentiles: Sequence[float],
) -> None:
    summary_raw = compute_summary_mean(
        y_load_path=y_load_path,
        chunk_size=chunk_size,
    )
    vesselness, mask_initial, mask_clean = build_vessel_outputs(
        summary_raw=summary_raw,
        frangi_sigmas=frangi_sigmas,
        frangi_black_ridges=frangi_black_ridges,
        mask_percentile=mask_percentile,
        min_object_area=min_object_area,
        mask_close_radius=mask_close_radius,
        mask_dilation_radius=mask_dilation_radius,
        input_percentiles=input_percentiles,
    )
    save_result_images(
        result_save_fold=result_save_fold,
        vesselness=vesselness,
        mask_initial=mask_initial,
        mask_clean=mask_clean,
    )


class Vessel:
    def __init__(
        self,
        y_load_path: str,
        result_save_fold: str,
        chunk_size: int,
        frangi_sigmas: Sequence[float],
        frangi_black_ridges: bool,
        mask_percentile: float,
        min_object_area: int,
        mask_close_radius: int,
        mask_dilation_radius: int,
        input_percentiles: Sequence[float],
        **kwargs,
    ) -> None:
        self.y_load_path = str(y_load_path)
        self.result_save_fold = str(result_save_fold)
        self.chunk_size = int(chunk_size)
        self.frangi_sigmas = tuple(float(x) for x in frangi_sigmas)
        self.frangi_black_ridges = bool(frangi_black_ridges)
        self.mask_percentile = float(mask_percentile)
        self.min_object_area = int(min_object_area)
        self.mask_close_radius = int(mask_close_radius)
        self.mask_dilation_radius = int(mask_dilation_radius)
        self.input_percentiles = tuple(float(x) for x in input_percentiles)

    def forward(self) -> None:
        run(
            y_load_path=self.y_load_path,
            result_save_fold=self.result_save_fold,
            chunk_size=self.chunk_size,
            frangi_sigmas=self.frangi_sigmas,
            frangi_black_ridges=self.frangi_black_ridges,
            mask_percentile=self.mask_percentile,
            min_object_area=self.min_object_area,
            mask_close_radius=self.mask_close_radius,
            mask_dilation_radius=self.mask_dilation_radius,
            input_percentiles=self.input_percentiles,
        )
