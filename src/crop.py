from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from pathlib import Path
import tifffile
import tqdm
import zarr

from .recon.stitch import Stitch


@dataclass(frozen=True)
class PatchSpec:
    quadrant_row: int
    quadrant_col: int
    patch_row: int
    patch_col: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def name(self) -> str:
        return (
            f"{self.quadrant_row}-{self.quadrant_col}-"
            f"{self.patch_row}-{self.patch_col}"
        )


def stripe_bounds(position: int, limit: int, width: int) -> tuple[int, int]:
    if position <= 0:
        return 0, min(width, limit)
    if position >= limit:
        return max(0, limit - width), limit

    left = position - (width // 2)
    right = left + width
    if left < 0:
        left = 0
        right = min(width, limit)
    if right > limit:
        right = limit
        left = max(0, limit - width)
    return left, right


def stripe_bounds_center(center: float, limit: int, width: int) -> tuple[int, int]:
    if width <= 0:
        return 0, 0
    left = int(np.floor(center - (width - 1) / 2.0))
    right = left + width
    left = max(0, left)
    right = min(limit, right)
    if right <= left:
        return 0, 0
    return left, right


def paint_quadrant_grid(
    mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    tile_size: int,
    seam_width: int,
) -> None:
    edge_width = seam_width // 2
    h = y1 - y0
    w = x1 - x0

    # Internal vertical seams (width 4), clipped inside this quadrant.
    for local_x in range(tile_size, w, tile_size):
        pos = x0 + local_x
        left, right = stripe_bounds(pos, mask.shape[1], seam_width)
        left = max(left, x0)
        right = min(right, x1)
        if left < right:
            mask[y0:y1, left:right] = True

    # Internal horizontal seams (width 4), clipped inside this quadrant.
    for local_y in range(tile_size, h, tile_size):
        pos = y0 + local_y
        top, bottom = stripe_bounds(pos, mask.shape[0], seam_width)
        top = max(top, y0)
        bottom = min(bottom, y1)
        if top < bottom:
            mask[top:bottom, x0:x1] = True

    # Boundary seams (width 2), always inside this quadrant.
    mask[y0:y1, x0:min(x0 + edge_width, x1)] = True
    mask[y0:y1, max(x1 - edge_width, x0):x1] = True
    mask[y0:min(y0 + edge_width, y1), x0:x1] = True
    mask[max(y1 - edge_width, y0):y1, x0:x1] = True


def build_quadrant_masks(
    height: int,
    width: int,
    tile_size: int,
    seam_width: int,
) -> list[np.ndarray]:
    y_mid = height // 2
    x_mid = width // 2
    quads = [
        (0, y_mid, 0, x_mid),          # top-left
        (0, y_mid, x_mid, width),      # top-right
        (y_mid, height, 0, x_mid),     # bottom-left
        (y_mid, height, x_mid, width), # bottom-right
    ]

    masks: list[np.ndarray] = []
    for y0, y1, x0, x1 in quads:
        mask = np.zeros((height, width), dtype=bool)
        paint_quadrant_grid(
            mask=mask,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            tile_size=tile_size,
            seam_width=seam_width,
        )
        masks.append(mask)
    return masks


def read_first_frame(path: Path) -> np.ndarray:
    data = tifffile.memmap(path)
    if data.ndim == 2:
        frame = np.asarray(data)
    elif data.ndim >= 3:
        frame = np.asarray(data[0])
    else:
        raise ValueError(f"Unsupported TIFF ndim: {data.ndim}")
    if frame.ndim != 2:
        raise ValueError(f"First frame is not 2D: {frame.shape}")
    return frame


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    output = frame.astype(np.float32, copy=False)
    finite = np.isfinite(output)
    rgb = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    if not np.any(finite):
        return rgb

    lo = np.percentile(output[finite], 1.0)
    hi = np.percentile(output[finite], 99.0)
    if hi <= lo:
        hi = lo + 1.0

    scaled = np.clip((output - lo) / (hi - lo), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    return rgb


def overlay_quadrant_grid(
    frame: np.ndarray,
    masks: list[np.ndarray],
) -> np.ndarray:
    rgb = frame_to_rgb(frame)
    colors = np.array(
        [
            [255, 64, 64],   # top-left
            [255, 200, 40],  # top-right
            [64, 196, 96],   # bottom-left
            [64, 128, 255],  # bottom-right
        ],
        dtype=np.uint8,
    )
    for idx, mask in enumerate(masks):
        rgb[mask] = colors[idx]
    return rgb


def overlay_white_lines(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = frame_to_rgb(frame)
    rgb[mask] = np.asarray((255, 255, 255), dtype=np.uint8)
    return rgb

def stitch_image(
    image: np.ndarray,
    match_point_1: tuple[tuple[int, int], tuple[int, int]],
    match_point_2: tuple[tuple[int, int], tuple[int, int]],
    match_point_3: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    stitch = Stitch(
        match_point_1=match_point_1,
        match_point_2=match_point_2,
        match_point_3=match_point_3,
        match_brightness=False,
        dtype="float32",
    )

    if image.ndim == 2:
        return stitch.forward(image.astype(np.float32))

    if image.ndim == 3 and image.shape[2] == 3:
        channels: list[np.ndarray] = []
        for c in range(3):
            stitched_c = stitch.forward(image[..., c].astype(np.float32))
            channels.append(stitched_c)
        stitched_rgb = np.stack(channels, axis=-1)
        return np.clip(stitched_rgb, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape for stitching: {image.shape}")


def parse_match_point(
    values: object, name: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        if (
            len(values) == 4
            and not isinstance(values[0], Sequence)
            and not isinstance(values[1], Sequence)
            and not isinstance(values[2], Sequence)
            and not isinstance(values[3], Sequence)
        ):
            return ((int(values[0]), int(values[1])), (int(values[2]), int(values[3])))
        if (
            len(values) == 2
            and isinstance(values[0], Sequence)
            and isinstance(values[1], Sequence)
            and len(values[0]) == 2
            and len(values[1]) == 2
        ):
            return (
                (int(values[0][0]), int(values[0][1])),
                (int(values[1][0]), int(values[1][1])),
            )
    raise ValueError(
        f"{name} must be [x1, y1, x2, y2] or [[x1, y1], [x2, y2]], got {values}"
    )


def matlab_range(start: int, end: int) -> np.ndarray:
    if end < start:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start - 1, end, dtype=np.int64)


def horizontal_stitch_metrics(
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    match_point: tuple[tuple[int, int], tuple[int, int]],
) -> dict[str, float]:
    h_a, w_a = shape_a
    h_b, w_b = shape_b
    mp = np.asarray(match_point, dtype=np.int64)

    shift_x = int(w_a - mp[0, 0] + mp[1, 0])
    shift_y = int(mp[1, 1] - mp[0, 1])
    if shift_x <= 0:
        raise ValueError(f"Invalid horizontal shift_x={shift_x}")

    img_a_overlap_y = matlab_range(max(1 - shift_y, 1), min(h_b - shift_y, h_a))
    if img_a_overlap_y.size <= 0:
        raise ValueError("Horizontal stitch has no Y overlap.")

    out_h = int(img_a_overlap_y.size)
    out_w = int(w_a + w_b - shift_x)
    seam_center_x = (w_a - shift_x) + (shift_x - 1) / 2.0
    return {
        "out_h": float(out_h),
        "out_w": float(out_w),
        "seam_center_x": float(seam_center_x),
    }


def vertical_stitch_metrics(
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    match_point: tuple[tuple[int, int], tuple[int, int]],
) -> dict[str, float]:
    h_a, w_a = shape_a
    h_b, w_b = shape_b
    mp = np.asarray(match_point, dtype=np.int64)

    shift_x = int(mp[1, 0] - mp[0, 0])
    shift_y = int(h_a - mp[0, 1] + mp[1, 1])
    if shift_y <= 0:
        raise ValueError(f"Invalid vertical shift_y={shift_y}")

    img_a_overlap_x = matlab_range(max(1 - shift_x, 1), min(w_b - shift_x, w_a))
    img_b_overlap_x = matlab_range(max(1 + shift_x, 1), min(w_a + shift_x, w_b))
    img_a_overlap_y = matlab_range(h_a - shift_y + 1, h_a)

    if img_a_overlap_x.size <= 0 or img_b_overlap_x.size <= 0:
        raise ValueError("Vertical stitch has no X overlap.")
    if img_a_overlap_y.size <= 0:
        raise ValueError("Vertical stitch has no Y overlap.")

    out_h = int(h_a + h_b - img_a_overlap_y.size)
    out_w = int(img_a_overlap_x.size)
    overlap_start_y = h_a - int(img_a_overlap_y.size)
    seam_center_y = overlap_start_y + (img_a_overlap_y.size - 1) / 2.0
    return {
        "out_h": float(out_h),
        "out_w": float(out_w),
        "a_x_start": float(int(img_a_overlap_x[0])),
        "b_x_start": float(int(img_b_overlap_x[0])),
        "seam_center_y": float(seam_center_y),
    }


def compute_midpoint_seam_geometry(
    input_shape: tuple[int, int],
    match_point_1: tuple[tuple[int, int], tuple[int, int]],
    match_point_2: tuple[tuple[int, int], tuple[int, int]],
    match_point_3: tuple[tuple[int, int], tuple[int, int]],
) -> dict[str, float]:
    h_in, w_in = input_shape
    if h_in % 2 != 0 or w_in % 2 != 0:
        raise ValueError("Input shape must be even in both dimensions.")

    half_h = h_in // 2
    half_w = w_in // 2
    top = horizontal_stitch_metrics((half_h, half_w), (half_h, half_w), match_point_1)
    bottom = horizontal_stitch_metrics(
        (half_h, half_w), (half_h, half_w), match_point_2
    )
    vertical = vertical_stitch_metrics(
        (int(top["out_h"]), int(top["out_w"])),
        (int(bottom["out_h"]), int(bottom["out_w"])),
        match_point_3,
    )
    return {
        "seam_top_x": float(top["seam_center_x"] - vertical["a_x_start"]),
        "seam_bottom_x": float(bottom["seam_center_x"] - vertical["b_x_start"]),
        "seam_mid_y": float(vertical["seam_center_y"]),
    }


def add_midpoint_stitch_seams(
    stitched: np.ndarray,
    input_shape: tuple[int, int],
    match_point_1: tuple[tuple[int, int], tuple[int, int]],
    match_point_2: tuple[tuple[int, int], tuple[int, int]],
    match_point_3: tuple[tuple[int, int], tuple[int, int]],
    seam_width: int,
) -> np.ndarray:
    seam_geom = compute_midpoint_seam_geometry(
        input_shape=input_shape,
        match_point_1=match_point_1,
        match_point_2=match_point_2,
        match_point_3=match_point_3,
    )
    seam_top_x = seam_geom["seam_top_x"]
    seam_bottom_x = seam_geom["seam_bottom_x"]
    seam_mid_y = seam_geom["seam_mid_y"]

    out = np.asarray(stitched).copy()
    out_h, out_w = out.shape[:2]
    y_split = int(np.floor(seam_mid_y))

    y0, y1 = stripe_bounds_center(seam_mid_y, out_h, seam_width)
    x0, x1 = stripe_bounds_center(seam_top_x, out_w, seam_width)
    xb0, xb1 = stripe_bounds_center(seam_bottom_x, out_w, seam_width)

    if out.ndim == 2:
        finite = np.isfinite(out)
        line_value = np.percentile(out[finite], 99.9) if np.any(finite) else 1.0
        if y0 < y1:
            out[y0:y1, :] = line_value
        if x0 < x1:
            out[:max(0, min(y_split + 1, out_h)), x0:x1] = line_value
        if xb0 < xb1:
            out[max(0, min(y_split + 1, out_h)):, xb0:xb1] = line_value
        return out

    if out.ndim == 3 and out.shape[2] == 3:
        color = np.asarray((255, 255, 255), dtype=np.uint8)
        if y0 < y1:
            out[y0:y1, :, :] = color
        if x0 < x1:
            out[:max(0, min(y_split + 1, out_h)), x0:x1, :] = color
        if xb0 < xb1:
            out[max(0, min(y_split + 1, out_h)):, xb0:xb1, :] = color
        return out

    raise ValueError(f"Unsupported stitched image shape: {out.shape}")


def find_true_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask_1d.astype(bool)):
        if v and not in_run:
            start = i
            in_run = True
        elif (not v) and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, int(mask_1d.size)))
    return runs


def merge_runs(runs: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for s, e in runs:
        s2 = max(0, min(limit, int(s)))
        e2 = max(0, min(limit, int(e)))
        if e2 > s2:
            clipped.append((s2, e2))
    if not clipped:
        return []

    clipped.sort(key=lambda p: (p[0], p[1]))
    merged: list[tuple[int, int]] = [clipped[0]]
    for s, e in clipped[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def center_to_cut(center: float, limit: int) -> int:
    if limit <= 1:
        return 0
    cut = int(np.floor(center + 0.5))
    return max(1, min(limit - 1, cut))


def runs_to_cut_positions(
    runs: list[tuple[int, int]],
    length: int,
    edge_margin: int,
) -> list[int]:
    cuts: list[int] = []
    for s, e in runs:
        if e <= s:
            continue
        if s <= 0 or e >= length:
            # Boundary seam visualization should not create extra tiny patches.
            continue
        center = (s + e) / 2.0
        cut = center_to_cut(center, length)
        if cut <= edge_margin or cut >= length - edge_margin:
            continue
        cuts.append(cut)
    return sorted(set(cuts))


def spans_from_cuts(
    length: int,
    cuts: list[int],
    min_patch_size: int,
) -> list[tuple[int, int]]:
    boundaries = [0] + sorted(set(c for c in cuts if 0 < c < length)) + [length]
    if len(boundaries) <= 2:
        return [(0, length)] if length > 0 else []

    # Do not drop pixels: merge tiny spans into adjacent spans by removing cuts.
    # Quadrant seams are represented by boundaries 0 and length and are never removed.
    while True:
        lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        small_idx = next(
            (i for i, span_len in enumerate(lengths) if span_len < min_patch_size),
            None,
        )
        if small_idx is None:
            break
        if len(boundaries) <= 2:
            break

        can_merge_left = small_idx > 0
        can_merge_right = (small_idx + 1) < (len(boundaries) - 1)
        if not can_merge_left and not can_merge_right:
            break

        if can_merge_left and can_merge_right:
            left_len = boundaries[small_idx] - boundaries[small_idx - 1]
            right_len = boundaries[small_idx + 2] - boundaries[small_idx + 1]
            # Both sides are internal seams: remove the side that yields the
            # smaller merged span (merge with the smaller neighbor).
            merge_left = left_len <= right_len
        else:
            # Edge tiny span: only the internal seam can be removed; keep
            # quadrant seam at boundary 0/length untouched.
            merge_left = can_merge_left

        if merge_left:
            # Remove left boundary: merge previous span and small span.
            del boundaries[small_idx]
        else:
            # Remove right boundary: merge small span and next span.
            del boundaries[small_idx + 1]

    spans: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e > s:
            spans.append((s, e))
    return spans


def detect_axis_seam_runs(
    seam_mask: np.ndarray, axis: str, coverage_ratio: float
) -> list[tuple[int, int]]:
    if seam_mask.ndim != 2:
        raise ValueError("seam_mask must be 2D.")
    h, w = seam_mask.shape
    if h <= 0 or w <= 0:
        return []

    coverage_ratio = float(np.clip(coverage_ratio, 0.0, 1.0))
    if axis == "horizontal":
        counts = np.count_nonzero(seam_mask, axis=1)
        threshold = max(1, int(np.ceil(w * coverage_ratio)))
        return find_true_runs(counts >= threshold)
    if axis == "vertical":
        counts = np.count_nonzero(seam_mask, axis=0)
        threshold = max(1, int(np.ceil(h * coverage_ratio)))
        return find_true_runs(counts >= threshold)
    raise ValueError("axis must be 'horizontal' or 'vertical'.")


def build_patch_specs_for_stitched(
    stitched_shape: tuple[int, int],
    pre_stitch_shape: tuple[int, int],
    tile_size: int,
    seam_width: int,
    match_point_1: tuple[tuple[int, int], tuple[int, int]],
    match_point_2: tuple[tuple[int, int], tuple[int, int]],
    match_point_3: tuple[tuple[int, int], tuple[int, int]],
    coverage_ratio: float,
    min_patch_size: int,
) -> list[PatchSpec]:
    edge_width = seam_width // 2
    out_h, out_w = stitched_shape
    seam_geom = compute_midpoint_seam_geometry(
        input_shape=pre_stitch_shape,
        match_point_1=match_point_1,
        match_point_2=match_point_2,
        match_point_3=match_point_3,
    )
    y_cut = center_to_cut(seam_geom["seam_mid_y"], out_h)
    x_cut_top = center_to_cut(seam_geom["seam_top_x"], out_w)
    x_cut_bottom = center_to_cut(seam_geom["seam_bottom_x"], out_w)

    regions: dict[tuple[int, int], tuple[int, int, int, int]] = {
        (0, 0): (0, y_cut, 0, x_cut_top),
        (0, 1): (0, y_cut, x_cut_top, out_w),
        (1, 0): (y_cut, out_h, 0, x_cut_bottom),
        (1, 1): (y_cut, out_h, x_cut_bottom, out_w),
    }

    stitched_masks = [
        stitch_image(mask.astype(np.float32), match_point_1, match_point_2, match_point_3)
        for mask in build_quadrant_masks(
            pre_stitch_shape[0], pre_stitch_shape[1], tile_size, seam_width
        )
    ]
    stitched_masks_bool = [m > 1e-6 for m in stitched_masks]
    for m in stitched_masks_bool:
        if m.shape != (out_h, out_w):
            raise ValueError(
                f"Stitched seam mask shape mismatch: got {m.shape}, expected {(out_h, out_w)}"
            )

    patch_specs: list[PatchSpec] = []
    quadrant_order = [
        ((0, 0), 0),  # top-left
        ((0, 1), 1),  # top-right
        ((1, 0), 2),  # bottom-left
        ((1, 1), 3),  # bottom-right
    ]
    for (q_row, q_col), q_idx in quadrant_order:
        ry0, ry1, rx0, rx1 = regions[(q_row, q_col)]
        if ry1 <= ry0 or rx1 <= rx0:
            raise ValueError(
                f"Invalid quadrant region for {(q_row, q_col)}: "
                f"y=({ry0},{ry1}) x=({rx0},{rx1})"
            )
        region_mask = stitched_masks_bool[q_idx][ry0:ry1, rx0:rx1]
        hq, wq = region_mask.shape

        row_runs = detect_axis_seam_runs(region_mask, "horizontal", coverage_ratio)
        col_runs = detect_axis_seam_runs(region_mask, "vertical", coverage_ratio)
        row_runs = merge_runs(row_runs, hq)
        col_runs = merge_runs(col_runs, wq)

        row_cuts = runs_to_cut_positions(
            row_runs, hq, edge_margin=max(1, edge_width * 2)
        )
        col_cuts = runs_to_cut_positions(
            col_runs, wq, edge_margin=max(1, edge_width * 2)
        )

        row_spans = spans_from_cuts(hq, row_cuts, min_patch_size)
        col_spans = spans_from_cuts(wq, col_cuts, min_patch_size)
        if not row_spans or not col_spans:
            raise ValueError(
                f"No valid patch spans in quadrant {(q_row, q_col)}. "
                f"row_spans={row_spans}, col_spans={col_spans}"
            )

        for p_row, (py0, py1) in enumerate(row_spans):
            for p_col, (px0, px1) in enumerate(col_spans):
                patch_specs.append(
                    PatchSpec(
                        quadrant_row=q_row,
                        quadrant_col=q_col,
                        patch_row=p_row,
                        patch_col=p_col,
                        y0=ry0 + py0,
                        y1=ry0 + py1,
                        x0=rx0 + px0,
                        x1=rx0 + px1,
                    )
                )

    coverage = np.zeros((out_h, out_w), dtype=np.uint8)
    for spec in patch_specs:
        coverage[spec.y0 : spec.y1, spec.x0 : spec.x1] += 1
    min_cov = int(coverage.min())
    max_cov = int(coverage.max())
    if min_cov != 1 or max_cov != 1:
        raise ValueError(
            f"Patch specs must cover stitched image exactly once, got "
            f"coverage min={min_cov}, max={max_cov}"
        )
    return patch_specs


def build_patch_boundary_mask(
    stitched_shape: tuple[int, int],
    patch_specs: list[PatchSpec],
    seam_width: int,
) -> np.ndarray:
    out_h, out_w = stitched_shape
    edge_width = seam_width // 2
    mask = np.zeros((out_h, out_w), dtype=bool)

    for spec in patch_specs:
        if not (0 <= spec.y0 < spec.y1 <= out_h and 0 <= spec.x0 < spec.x1 <= out_w):
            raise ValueError(
                f"Patch {spec.name} out of bounds for boundary mask: "
                f"y=({spec.y0},{spec.y1}) x=({spec.x0},{spec.x1}) shape={(out_h, out_w)}"
            )
        mask[spec.y0 : min(spec.y0 + edge_width, spec.y1), spec.x0 : spec.x1] = True
        mask[max(spec.y1 - edge_width, spec.y0) : spec.y1, spec.x0 : spec.x1] = True
        mask[spec.y0 : spec.y1, spec.x0 : min(spec.x0 + edge_width, spec.x1)] = True
        mask[spec.y0 : spec.y1, max(spec.x1 - edge_width, spec.x0) : spec.x1] = True

    return mask


def extract_patches_from_stitched(
    stitched_tif: Path,
    patch_specs: list[PatchSpec],
    output_dir: Path,
    chunk_size: int,
    frame_start: int,
    frame_end: int,
    pad: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(stitched_tif) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            video = video[None, ...]
        if video.ndim != 3:
            raise ValueError(f"Expected 2D or 3D TIFF, got ndim={video.ndim}")

        total_frames = int(video.shape[0])
        h = int(video.shape[1])
        w = int(video.shape[2])
        for spec in patch_specs:
            if not (0 <= spec.y0 < spec.y1 <= h and 0 <= spec.x0 < spec.x1 <= w):
                raise ValueError(
                    f"Patch {spec.name} out of bounds: "
                    f"y=({spec.y0},{spec.y1}) x=({spec.x0},{spec.x1}) for {(h, w)}"
                )

        if frame_start < 0:
            raise ValueError("frame_start must be >= 0")
        if frame_end == -1:
            frame_end = total_frames
        frame_end = min(frame_end, total_frames)
        if frame_end <= frame_start:
            raise ValueError(
                f"Invalid frame range [{frame_start}, {frame_end}) for {total_frames} frames."
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if pad < 0:
            raise ValueError("pad must be >= 0")

        writers: dict[str, tifffile.TiffWriter] = {}
        try:
            for spec in patch_specs:
                out_path = output_dir / f"{spec.name}.tif"
                writers[spec.name] = tifffile.TiffWriter(out_path, bigtiff=True)

            total_chunks = (frame_end - frame_start + chunk_size - 1) // chunk_size
            with tqdm.tqdm(
                total=total_chunks,
                desc="crop",
                unit="chunk",
                dynamic_ncols=True,
            ) as pbar:
                for start in range(frame_start, frame_end, chunk_size):
                    end = min(start + chunk_size, frame_end)
                    chunk = np.asarray(video[start:end])
                    for spec in patch_specs:
                        if pad == 0:
                            sub = chunk[:, spec.y0 : spec.y1, spec.x0 : spec.x1]
                        else:
                            out_h = (spec.y1 - spec.y0) + 2 * pad
                            out_w = (spec.x1 - spec.x0) + 2 * pad
                            sub = np.zeros(
                                (chunk.shape[0], out_h, out_w),
                                dtype=chunk.dtype,
                            )
                            sy0_req = spec.y0 - pad
                            sy1_req = spec.y1 + pad
                            sx0_req = spec.x0 - pad
                            sx1_req = spec.x1 + pad

                            sy0 = max(0, sy0_req)
                            sy1 = min(h, sy1_req)
                            sx0 = max(0, sx0_req)
                            sx1 = min(w, sx1_req)

                            dy0 = sy0 - sy0_req
                            dx0 = sx0 - sx0_req
                            dy1 = dy0 + (sy1 - sy0)
                            dx1 = dx0 + (sx1 - sx0)
                            sub[:, dy0:dy1, dx0:dx1] = chunk[:, sy0:sy1, sx0:sx1]
                        writer = writers[spec.name]
                        for frame in sub:
                            writer.write(
                                frame,
                                contiguous=True,
                                photometric="minisblack",
                            )
                    pbar.update(1)
        finally:
            for writer in writers.values():
                writer.close()


def save_image(path: Path, image: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        image.astype(np.uint8, copy=False),
        photometric="rgb",
        metadata={"axes": "YXS"},
    )


def run(
    x_load_path: str,
    y_load_path: str,
    y_save_fold: str | None,
    frame: Sequence[int],
    chunk_size: int,
    seam_width: int,
    tile_size: int,
    match_point_1: object,
    match_point_2: object,
    match_point_3: object,
    coverage_ratio: float,
    min_patch_size: int,
    pad: int,
) -> None:
    input_path = Path(x_load_path)
    if y_load_path is None:
        raise ValueError("y_load_path must be set.")
    patch_input = Path(y_load_path)

    first_frame = read_first_frame(input_path)
    h, w = first_frame.shape

    mp1 = parse_match_point(match_point_1, "match_point_1")
    mp2 = parse_match_point(match_point_2, "match_point_2")
    mp3 = parse_match_point(match_point_3, "match_point_3")
    seam_width = int(seam_width)
    if seam_width < 2 or seam_width % 2 != 0:
        raise ValueError(
            f"seam_width must be an even integer >= 2, got {seam_width}"
        )
    tile_size = int(tile_size)

    with tifffile.TiffFile(patch_input) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            stitched_shape = (int(video.shape[0]), int(video.shape[1]))
        elif video.ndim == 3:
            stitched_shape = (int(video.shape[1]), int(video.shape[2]))
        else:
            raise ValueError(
                f"Unsupported stitched TIFF ndim={video.ndim}: {patch_input}"
            )

    patch_specs = build_patch_specs_for_stitched(
        stitched_shape=stitched_shape,
        pre_stitch_shape=(h, w),
        tile_size=tile_size,
        seam_width=seam_width,
        match_point_1=mp1,
        match_point_2=mp2,
        match_point_3=mp3,
        coverage_ratio=float(coverage_ratio),
        min_patch_size=int(min_patch_size),
    )

    output_dir = (
        Path(y_save_fold)
        if y_save_fold is not None
        else patch_input.with_suffix("")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_cfg = frame
    if len(frame_cfg) != 2:
        raise ValueError("frame must be [start, end], end=-1 means full.")
    frame_start = int(frame_cfg[0])
    frame_end = int(frame_cfg[1])

    extract_patches_from_stitched(
        stitched_tif=patch_input,
        patch_specs=patch_specs,
        output_dir=output_dir,
        chunk_size=int(chunk_size),
        frame_start=frame_start,
        frame_end=frame_end,
        pad=int(pad),
    )


class Crop:
    def __init__(
        self,
        x_load_path: str,
        y_load_path: str,
        y_save_fold: str | None,
        frame: Sequence[int],
        chunk_size: int,
        seam_width: int,
        tile_size: int,
        match_point_1: object,
        match_point_2: object,
        match_point_3: object,
        coverage_ratio: float,
        min_patch_size: int,
        pad: int,
        **kwargs,
    ) -> None:
        self.x_load_path = x_load_path
        self.y_load_path = y_load_path
        self.y_save_fold = y_save_fold
        self.frame = list(frame)
        self.chunk_size = int(chunk_size)
        self.seam_width = int(seam_width)
        self.tile_size = int(tile_size)
        self.match_point_1 = match_point_1
        self.match_point_2 = match_point_2
        self.match_point_3 = match_point_3
        self.coverage_ratio = float(coverage_ratio)
        self.min_patch_size = int(min_patch_size)
        self.pad = int(pad)

    def forward(self) -> None:
        run(
            x_load_path=self.x_load_path,
            y_load_path=self.y_load_path,
            y_save_fold=self.y_save_fold,
            frame=self.frame,
            chunk_size=self.chunk_size,
            seam_width=self.seam_width,
            tile_size=self.tile_size,
            match_point_1=self.match_point_1,
            match_point_2=self.match_point_2,
            match_point_3=self.match_point_3,
            coverage_ratio=self.coverage_ratio,
            min_patch_size=self.min_patch_size,
            pad=self.pad,
        )
