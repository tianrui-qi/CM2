from __future__ import annotations
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import tifffile
import tqdm


POINTMAP_DIRNAME = "figure-PointMap"
POINTMAP_LABEL_DIRNAME = "figure-PointMap-label"
THRESHOLDSTACK_DIRNAME = "figure-ThresholdStack"

OVERLAP_POINTMAP_DIRNAME = "overlap-PointMap"
OVERLAP_POINTMAP_LABEL_DIRNAME = "overlap-PointMap-label"
OVERLAP_THRESHOLDSTACK_DIRNAME = "overlap-ThresholdStack"

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


def _background_to_rgb_uint8(background: np.ndarray) -> np.ndarray:
    if background.ndim == 2:
        gray = _normalize_to_uint8(background)
        return np.repeat(gray[..., None], 3, axis=-1)
    if background.ndim == 3 and background.shape[-1] in (3, 4):
        rgb = background[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        if rgb.dtype == np.uint16:
            return np.round(rgb.astype(np.float32) / 257.0).astype(np.uint8)
        channels = [
            _normalize_to_uint8(rgb[..., channel_idx]) for channel_idx in range(3)
        ]
        return np.stack(channels, axis=-1)
    raise ValueError(
        f"Background must be 2D grayscale or YXS RGB/RGBA, got shape={background.shape}"
    )


def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    if src_h == target_h and src_w == target_w:
        return image
    y_idx = np.minimum(
        (np.arange(target_h, dtype=np.int64) * src_h) // target_h, src_h - 1
    )
    x_idx = np.minimum(
        (np.arange(target_w, dtype=np.int64) * src_w) // target_w, src_w - 1
    )
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
    for tif_path in path.glob("*.tif"):
        tif_path.unlink(missing_ok=True)


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
            tifffile.imwrite(
                output_dir / source_path.name,
                out,
                photometric="rgb",
            )
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

        raise ValueError(
            f"Unsupported foreground shape for {source_path}: {foreground.shape}"
        )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="pipeline/overlap",
)
def main(cfg: omegaconf.DictConfig) -> None:
    background_path = Path(cfg.background_path).expanduser().resolve()
    extract_dir = Path(cfg.extract_path).expanduser().resolve()

    if not background_path.is_file():
        raise FileNotFoundError(f"Background not found: {background_path}")
    if not extract_dir.is_dir():
        raise NotADirectoryError(f"Extract folder not found: {extract_dir}")

    background = _read_tiff(background_path)
    background_rgb_base = _background_to_rgb_uint8(background)

    pointmap_dir = extract_dir / POINTMAP_DIRNAME
    pointmap_label_dir = extract_dir / POINTMAP_LABEL_DIRNAME
    thresholdstack_dir = extract_dir / THRESHOLDSTACK_DIRNAME
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
        dynamic_ncols=True,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        unit="figure",
    ) as progress_bar:
        _overlay_folder(
            pointmap_dir,
            extract_dir / OVERLAP_POINTMAP_DIRNAME,
            background_rgb_base,
            preserve_labels=False,
            progress_bar=progress_bar,
        )
        _overlay_folder(
            pointmap_label_dir,
            extract_dir / OVERLAP_POINTMAP_LABEL_DIRNAME,
            background_rgb_base,
            preserve_labels=False,
            progress_bar=progress_bar,
        )
        _overlay_folder(
            thresholdstack_dir,
            extract_dir / OVERLAP_THRESHOLDSTACK_DIRNAME,
            background_rgb_base,
            preserve_labels=True,
            progress_bar=progress_bar,
        )


if __name__ == "__main__":
    main()
