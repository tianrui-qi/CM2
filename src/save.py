from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
import re
import shutil
import tempfile

import numpy as np
import tifffile
import zarr

from .crop import (
    add_midpoint_stitch_seams,
    build_patch_boundary_mask,
    build_patch_specs_for_stitched,
    build_quadrant_masks,
    overlay_quadrant_grid,
    overlay_white_lines,
    parse_match_point,
    read_first_frame,
    save_image,
    stitch_image,
)


PATCH_NAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)$")


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
    ds_ly0: int = 0
    ds_lx0: int = 0
    ds_y0: int = 0
    ds_y1: int = 0
    ds_x0: int = 0
    ds_x1: int = 0


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
    model_load_fold_cfg: str | None,
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

    model_load_fold = (
        y_load_path.parent / "model"
        if model_load_fold_cfg is None
        else Path(model_load_fold_cfg)
    )
    if not model_load_fold.is_dir():
        raise NotADirectoryError(
            f"model_load_fold must be an existing directory: {model_load_fold}"
        )
    return y_load_fold, model_load_fold


def _collect_patch_meta(
    y_load_fold: Path,
    model_load_fold: Path,
) -> list[PatchMeta]:
    patches: list[PatchMeta] = []
    model_files = sorted(model_load_fold.glob("*.hdf5"), key=lambda p: p.name)
    if not model_files:
        raise FileNotFoundError(f"No .hdf5 model files found in: {model_load_fold}")

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


def _prepare_patch_downsample_slices(p: PatchMeta, factor: int) -> None:
    off_y = (-p.y0) % factor
    off_x = (-p.x0) % factor
    p.ds_ly0 = p.core_ly0 + off_y
    p.ds_lx0 = p.core_lx0 + off_x
    p.ds_y0 = (p.y0 + off_y) // factor
    p.ds_x0 = (p.x0 + off_x) // factor
    h_ds = (p.core_h - off_y + factor - 1) // factor
    w_ds = (p.core_w - off_x + factor - 1) // factor
    p.ds_y1 = p.ds_y0 + h_ds
    p.ds_x1 = p.ds_x0 + w_ds


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


def _estimate_worker_memory_bytes(
    patches: list[PatchMeta],
    model_chunk: int,
) -> int:
    if not patches:
        return 1 << 30
    float_bytes = np.dtype(np.float32).itemsize
    max_pixels = max(p.h * p.w for p in patches)
    max_model_bytes = max(int(p.model_path.stat().st_size) for p in patches)
    dense_work = max_pixels * model_chunk * float_bytes * 6
    frame_buffers = max_pixels * float_bytes * 6
    model_overhead = int(max_model_bytes * 5 // 2)
    fixed_overhead = 512 * 1024 * 1024
    estimate = dense_work + frame_buffers + model_overhead + fixed_overhead
    return max(256 * 1024 * 1024, estimate)


def _resolve_parallel_workers(
    patches: list[PatchMeta],
    model_chunk: int,
    requested_workers: int | None,
    memory_usage_fraction: float,
) -> int:
    if not patches:
        return 1
    cpu_total = max(1, os.cpu_count() or 1)
    cpu_cap = max(1, cpu_total - 1)

    if requested_workers is not None:
        req = int(requested_workers)
        if req <= 0:
            raise ValueError(f"model_workers must be > 0 when set, got {req}")
        target = min(req, cpu_cap)
    else:
        target = cpu_cap

    frac = float(memory_usage_fraction)
    if not (0.1 <= frac <= 0.95):
        raise ValueError(
            f"model_memory_fraction must be in [0.1, 0.95], got {memory_usage_fraction}"
        )

    avail_bytes = _get_available_memory_bytes()
    memory_cap = len(patches)
    if avail_bytes is not None:
        worker_mem = _estimate_worker_memory_bytes(patches, model_chunk)
        budget = int(avail_bytes * frac)
        if budget <= 0:
            memory_cap = 1
        else:
            memory_cap = max(1, budget // max(worker_mem, 1))

    workers = min(len(patches), target, memory_cap)
    return max(1, workers)


def _process_single_patch_to_temp(
    p: PatchMeta,
    tmp_dir: Path,
    factor: int,
    model_chunk: int,
    bf_name: str,
    b_name: str,
    ac_name: str,
) -> tuple[str, int]:
    from caiman.source_extraction.cnmf.cnmf import load_CNMF

    logging.getLogger("caiman").setLevel(logging.ERROR)
    patch_dir = tmp_dir / p.patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)

    cnm = load_CNMF(str(p.model_path), n_processes=1, dview=None)
    dims = tuple(int(x) for x in list(cnm.dims)[:2])
    if dims != (p.h, p.w):
        raise ValueError(
            f"Movie/model dims mismatch for {p.patch_name}: movie={(p.h, p.w)}, model={dims}"
        )

    A = cnm.estimates.A.tocsr()
    C = np.asarray(cnm.estimates.C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"Expected C as 2D matrix, got shape={C.shape}")
    if A.shape[0] != p.h * p.w or A.shape[1] != C.shape[0]:
        raise ValueError(
            f"A/C shape mismatch for {p.patch_name}: A={A.shape}, C={C.shape}"
        )

    b0_raw = getattr(cnm.estimates, "b0", None)
    if b0_raw is None:
        raise ValueError(f"Model has no b0 for {p.patch_name}: {p.model_path}")
    b0 = np.asarray(b0_raw, dtype=np.float32).reshape(-1)
    if b0.size != p.h * p.w:
        raise ValueError(
            f"b0 size mismatch for {p.patch_name}: {b0.size} vs {p.h * p.w}"
        )

    W = getattr(cnm.estimates, "W", None)
    if W is None:
        raise ValueError(f"Model has no W for {p.patch_name}: {p.model_path}")

    t_use = min(p.t, int(C.shape[1]))
    if t_use <= 0:
        raise ValueError(f"No valid frames for patch: {p.patch_name}")

    bc_patch = b0.reshape((p.h, p.w), order="F").astype(np.float32, copy=False)
    asum_patch = np.asarray(A.sum(axis=1), dtype=np.float32).reshape(-1)
    asum_patch = asum_patch.reshape((p.h, p.w), order="F").astype(
        np.float32, copy=False
    )
    csum = C[:, :t_use].sum(axis=1, dtype=np.float32)
    acsum_patch = np.asarray(A @ csum, dtype=np.float32).reshape(-1)
    acsum_patch = acsum_patch.reshape((p.h, p.w), order="F").astype(
        np.float32, copy=False
    )
    bc_core = bc_patch[p.core_ly0 : p.core_ly1, p.core_lx0 : p.core_lx1]
    asum_core = asum_patch[p.core_ly0 : p.core_ly1, p.core_lx0 : p.core_lx1]
    acsum_core = acsum_patch[p.core_ly0 : p.core_ly1, p.core_lx0 : p.core_lx1]
    bc_down = bc_patch[
        p.ds_ly0 : p.core_ly1 : factor,
        p.ds_lx0 : p.core_lx1 : factor,
    ]

    tifffile.imwrite(
        patch_dir / "Bc.tif", bc_core, imagej=True, metadata={"axes": "YX"}
    )
    tifffile.imwrite(
        patch_dir / "Asum.tif",
        asum_core,
        imagej=True,
        metadata={"axes": "YX"},
    )
    tifffile.imwrite(
        patch_dir / "ACsum.tif",
        acsum_core,
        imagej=True,
        metadata={"axes": "YX"},
    )

    with tifffile.TiffFile(p.movie_path) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim != 3:
            raise ValueError(f"Patch movie must be TYX, got ndim={video.ndim}: {p.movie_path}")
        if int(video.shape[0]) < t_use:
            raise ValueError(
                f"Patch movie shorter than expected for {p.patch_name}: {video.shape[0]} < {t_use}"
            )

        with tifffile.TiffWriter(
            patch_dir / bf_name,
            bigtiff=True,
        ) as writer_bf, tifffile.TiffWriter(
            patch_dir / b_name,
            bigtiff=True,
        ) as writer_b, tifffile.TiffWriter(
            patch_dir / ac_name,
            bigtiff=True,
        ) as writer_ac:
            for start in range(0, t_use, model_chunk):
                end = min(start + model_chunk, t_use)
                tc = end - start
                block = np.asarray(video[start:end], dtype=np.float32)
                y_mat = np.transpose(block, (1, 2, 0)).reshape(
                    (p.h * p.w, tc), order="F"
                )

                c_chunk = C[:, start:end]
                ac_mat = np.asarray(A @ c_chunk, dtype=np.float32)
                residual = y_mat - ac_mat - b0[:, None]
                bf_mat = np.asarray(W @ residual, dtype=np.float32)

                ac_frames = ac_mat.T.reshape((tc, p.h, p.w), order="F")
                bf_frames = bf_mat.T.reshape((tc, p.h, p.w), order="F")

                ac_down = ac_frames[
                    :,
                    p.ds_ly0 : p.core_ly1 : factor,
                    p.ds_lx0 : p.core_lx1 : factor,
                ]
                bf_down = bf_frames[
                    :,
                    p.ds_ly0 : p.core_ly1 : factor,
                    p.ds_lx0 : p.core_lx1 : factor,
                ]
                b_down = bf_down + bc_down[None, :, :]

                for i in range(tc):
                    writer_bf.write(
                        bf_down[i].astype(np.float32, copy=False),
                        contiguous=True,
                        photometric="minisblack",
                    )
                    writer_b.write(
                        b_down[i].astype(np.float32, copy=False),
                        contiguous=True,
                        photometric="minisblack",
                    )
                    writer_ac.write(
                        ac_down[i].astype(np.float32, copy=False),
                        contiguous=True,
                        photometric="minisblack",
                    )

    return p.patch_name, int(t_use)


def _stitch_video_from_temp(
    patches: list[PatchMeta],
    tmp_root: Path,
    patch_video_name: str,
    out_path: Path,
    full_h_ds: int,
    full_w_ds: int,
    global_t: int,
) -> None:
    opened: list[tifffile.TiffFile] = []
    arrays: list[zarr.Array] = []
    try:
        for p in patches:
            patch_file = tmp_root / p.patch_name / patch_video_name
            tf = tifffile.TiffFile(patch_file)
            opened.append(tf)
            arr = zarr.open(tf.aszarr(), mode="r")
            if arr.ndim != 3:
                raise ValueError(f"Expected TYX patch video, got ndim={arr.ndim}: {patch_file}")
            if int(arr.shape[0]) < global_t:
                raise ValueError(
                    f"Patch video shorter than expected for {p.patch_name}: {arr.shape[0]} < {global_t}"
                )
            arrays.append(arr)

        with tifffile.TiffWriter(out_path, bigtiff=True) as writer:
            for t in range(global_t):
                frame = np.zeros((full_h_ds, full_w_ds), dtype=np.float32)
                for p, arr in zip(patches, arrays):
                    patch_frame = np.asarray(arr[t], dtype=np.float32)
                    frame[p.ds_y0 : p.ds_y1, p.ds_x0 : p.ds_x1] = patch_frame
                writer.write(
                    frame,
                    contiguous=True,
                    photometric="minisblack",
                )
    finally:
        for tf in opened:
            tf.close()


def save_model_products_stitched(
    y_load_path: Path,
    model_load_fold_cfg: str | None,
    core_specs: dict[str, tuple[int, int, int, int]],
    save_dir: Path,
    downsample_factor: int,
    chunk_size: int,
    model_workers: int | None,
    model_memory_fraction: float,
) -> None:
    factor = int(downsample_factor)
    if factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {factor}")
    bf_name = "Bf.tif" if factor == 1 else f"Bf-d{factor}.tif"
    b_name = "B.tif" if factor == 1 else f"B-d{factor}.tif"
    ac_name = "AC.tif" if factor == 1 else f"AC-d{factor}.tif"
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk}")
    # Keep model-product chunk small to control peak memory.
    model_chunk = max(1, min(chunk, 4))

    y_load_fold, model_load_fold = _resolve_patch_folders(y_load_path, model_load_fold_cfg)
    patches = _collect_patch_meta(y_load_fold, model_load_fold)
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

    if (full_h % factor) != 0 or (full_w % factor) != 0:
        raise ValueError(
            f"downsample_factor={factor} must divide stitched H/W exactly, got {(full_h, full_w)}"
        )
    full_h_ds = full_h // factor
    full_w_ds = full_w // factor

    for p in patches:
        _prepare_patch_downsample_slices(p, factor)

    # Silence verbose CaImAn parameter-diff logs during model loading.
    logging.getLogger("caiman").setLevel(logging.ERROR)

    tmp_dir = Path(tempfile.mkdtemp(prefix="save_model_tmp_", dir=save_dir))
    try:
        workers = _resolve_parallel_workers(
            patches=patches,
            model_chunk=model_chunk,
            requested_workers=model_workers,
            memory_usage_fraction=float(model_memory_fraction),
        )

        # Phase 1: compute each patch result and save locally (memory-friendly).
        t_use_by_patch: dict[str, int] = {}
        if workers <= 1:
            for p in patches:
                patch_name, t_use = _process_single_patch_to_temp(
                    p=p,
                    tmp_dir=tmp_dir,
                    factor=factor,
                    model_chunk=model_chunk,
                    bf_name=bf_name,
                    b_name=b_name,
                    ac_name=ac_name,
                )
                t_use_by_patch[patch_name] = t_use
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _process_single_patch_to_temp,
                        p,
                        tmp_dir,
                        factor,
                        model_chunk,
                        bf_name,
                        b_name,
                        ac_name,
                    )
                    for p in patches
                ]
                for future in as_completed(futures):
                    patch_name, t_use = future.result()
                    t_use_by_patch[patch_name] = t_use

        for p in patches:
            if p.patch_name not in t_use_by_patch:
                raise RuntimeError(f"Missing t_use result for patch: {p.patch_name}")
            p.t = int(t_use_by_patch[p.patch_name])

        # Phase 2: stitch all patch results into final full-FOV outputs.
        global_t = min(p.t for p in patches)
        if global_t <= 0:
            raise ValueError("No frames available after per-patch model processing.")

        for patch_name, out_name in (
            ("Bc.tif", "Bc.tif"),
            ("Asum.tif", "Asum.tif"),
            ("ACsum.tif", "ACsum.tif"),
        ):
            full_2d = np.zeros((full_h, full_w), dtype=np.float32)
            for p in patches:
                patch_dir = tmp_dir / p.patch_name
                patch_2d = tifffile.imread(patch_dir / patch_name).astype(
                    np.float32, copy=False
                )
                full_2d[p.y0 : p.y1, p.x0 : p.x1] = patch_2d
            tifffile.imwrite(
                save_dir / out_name,
                full_2d,
                imagej=True,
                metadata={"axes": "YX"},
            )

        _stitch_video_from_temp(
            patches=patches,
            tmp_root=tmp_dir,
            patch_video_name=bf_name,
            out_path=save_dir / bf_name,
            full_h_ds=full_h_ds,
            full_w_ds=full_w_ds,
            global_t=global_t,
        )
        _stitch_video_from_temp(
            patches=patches,
            tmp_root=tmp_dir,
            patch_video_name=b_name,
            out_path=save_dir / b_name,
            full_h_ds=full_h_ds,
            full_w_ds=full_w_ds,
            global_t=global_t,
        )
        _stitch_video_from_temp(
            patches=patches,
            tmp_root=tmp_dir,
            patch_video_name=ac_name,
            out_path=save_dir / ac_name,
            full_h_ds=full_h_ds,
            full_w_ds=full_w_ds,
            global_t=global_t,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def save_downsampled_tif(
    x_path: Path,
    out_path: Path,
    downsample_factor: int,
    chunk_size: int,
) -> None:
    factor = int(downsample_factor)
    if factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {factor}")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk}")

    with tifffile.TiffFile(x_path) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            h = int(video.shape[0])
            w = int(video.shape[1])
            if (h % factor) != 0 or (w % factor) != 0:
                raise ValueError(
                    f"downsample_factor={factor} must divide H/W exactly, got {(h, w)}"
                )
            frame = np.asarray(video)
            frame_ds = frame[::factor, ::factor]
            with tifffile.TiffWriter(out_path, bigtiff=True) as writer:
                writer.write(
                    frame_ds,
                    contiguous=True,
                    photometric="minisblack",
                )
            return

        if video.ndim != 3:
            raise ValueError(f"Expected 2D or 3D TIFF, got ndim={video.ndim}: {x_path}")

        t = int(video.shape[0])
        h = int(video.shape[1])
        w = int(video.shape[2])
        if (h % factor) != 0 or (w % factor) != 0:
            raise ValueError(
                f"downsample_factor={factor} must divide H/W exactly, got {(h, w)}"
            )

        with tifffile.TiffWriter(out_path, bigtiff=True) as writer:
            for start in range(0, t, chunk):
                end = min(start + chunk, t)
                chunk_data = np.asarray(video[start:end])
                down = chunk_data[:, ::factor, ::factor]
                for frame in down:
                    writer.write(
                        frame,
                        contiguous=True,
                        photometric="minisblack",
                    )


def normalize_downsample_factors(raw: object) -> list[int]:
    if isinstance(raw, (int, np.integer)):
        factors = [int(raw)]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        factors = [int(x) for x in raw]
    else:
        raise ValueError(
            "downsample_factor must be an integer or a list of integers, "
            f"got {type(raw).__name__}"
        )

    if not factors:
        raise ValueError("downsample_factor list cannot be empty.")

    unique: list[int] = []
    seen: set[int] = set()
    for f in factors:
        if f <= 0:
            raise ValueError(f"downsample_factor values must be > 0, got {f}")
        if f in seen:
            continue
        seen.add(f)
        unique.append(f)
    return unique


def run(
    x_load_path: str,
    y_load_path: str,
    model_load_fold: str | None,
    save_fold: str | None,
    downsample_factor: object,
    chunk_size: int,
    seam_width: int,
    tile_size: int,
    match_point_1: object,
    match_point_2: object,
    match_point_3: object,
    coverage_ratio: float,
    min_patch_size: int,
    model_workers: int | None = None,
    model_memory_fraction: float = 0.55,
) -> None:
    input_path = Path(x_load_path)
    y_input_path = Path(y_load_path)
    plot_dir = (
        Path(save_fold)
        if save_fold is not None
        else input_path.parent / "save"
    )
    x_base = input_path.with_suffix("").name
    y_base = y_input_path.with_suffix("").name
    downsample_factors = normalize_downsample_factors(downsample_factor)
    save_path_1 = plot_dir / "crop-1.tif"
    save_path_2 = plot_dir / "crop-2.tif"
    save_path_3 = plot_dir / "crop-3.tif"
    save_path_4 = plot_dir / "crop-4.tif"
    save_path_first_frame_x = plot_dir / f"{x_base}-f0.tif"
    save_path_first_frame_y = plot_dir / f"{y_base}-f0.tif"

    first_frame = read_first_frame(input_path)
    first_frame_y = read_first_frame(y_input_path)
    h, w = first_frame.shape

    plot_dir.mkdir(parents=True, exist_ok=True)
    mp1 = parse_match_point(match_point_1, "match_point_1")
    mp2 = parse_match_point(match_point_2, "match_point_2")
    mp3 = parse_match_point(match_point_3, "match_point_3")
    seam_width = int(seam_width)
    if seam_width < 2 or seam_width % 2 != 0:
        raise ValueError(
            f"seam_width must be an even integer >= 2, got {seam_width}"
        )
    tile_size = int(tile_size)

    masks = build_quadrant_masks(
        height=h,
        width=w,
        tile_size=tile_size,
        seam_width=seam_width,
    )
    overlay_rgb = overlay_quadrant_grid(first_frame, masks)
    save_image(save_path_1, overlay_rgb)
    stitched_rgb = stitch_image(
        image=overlay_rgb,
        match_point_1=mp1,
        match_point_2=mp2,
        match_point_3=mp3,
    )
    stitched_rgb = add_midpoint_stitch_seams(
        stitched=stitched_rgb,
        input_shape=(h, w),
        match_point_1=mp1,
        match_point_2=mp2,
        match_point_3=mp3,
        seam_width=seam_width,
    )
    save_image(save_path_2, stitched_rgb)
    stitched_frame = stitch_image(
        image=first_frame,
        match_point_1=mp1,
        match_point_2=mp2,
        match_point_3=mp3,
    )
    stitched_shape = (int(stitched_frame.shape[0]), int(stitched_frame.shape[1]))

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
    core_specs = {
        spec.name: (int(spec.y0), int(spec.y1), int(spec.x0), int(spec.x1))
        for spec in patch_specs
    }
    crop_mask = build_patch_boundary_mask(
        stitched_shape=stitched_shape,
        patch_specs=patch_specs,
        seam_width=seam_width,
    )
    crop_overlay = overlay_white_lines(stitched_frame, crop_mask)
    save_image(save_path_3, crop_overlay)

    seam_only = np.zeros((stitched_shape[0], stitched_shape[1], 3), dtype=np.uint8)
    seam_only[crop_mask] = np.asarray((255, 255, 255), dtype=np.uint8)
    save_image(save_path_4, seam_only)
    tifffile.imwrite(save_path_first_frame_x, first_frame)
    tifffile.imwrite(save_path_first_frame_y, first_frame_y)
    for factor in downsample_factors:
        if factor > 1:
            save_path_downsample_x = plot_dir / f"{x_base}-d{factor}.tif"
            save_path_downsample_y = plot_dir / f"{y_base}-d{factor}.tif"
            save_downsampled_tif(
                x_path=input_path,
                out_path=save_path_downsample_x,
                downsample_factor=factor,
                chunk_size=int(chunk_size),
            )
            save_downsampled_tif(
                x_path=y_input_path,
                out_path=save_path_downsample_y,
                downsample_factor=factor,
                chunk_size=int(chunk_size),
            )

        save_model_products_stitched(
            y_load_path=y_input_path,
            model_load_fold_cfg=model_load_fold,
            core_specs=core_specs,
            save_dir=plot_dir,
            downsample_factor=factor,
            chunk_size=int(chunk_size),
            model_workers=model_workers,
            model_memory_fraction=float(model_memory_fraction),
        )


class Save:
    def __init__(
        self,
        x_load_path: str,
        y_load_path: str,
        model_load_fold: str | None,
        save_fold: str | None,
        downsample_factor: object,
        chunk_size: int,
        seam_width: int,
        tile_size: int,
        match_point_1: object,
        match_point_2: object,
        match_point_3: object,
        coverage_ratio: float,
        min_patch_size: int,
        model_workers: int | None = None,
        model_memory_fraction: float = 0.55,
        **kwargs,
    ) -> None:
        self.x_load_path = x_load_path
        self.y_load_path = y_load_path
        self.model_load_fold = model_load_fold
        self.save_fold = save_fold
        self.downsample_factor = downsample_factor
        self.chunk_size = int(chunk_size)
        self.seam_width = int(seam_width)
        self.tile_size = int(tile_size)
        self.match_point_1 = match_point_1
        self.match_point_2 = match_point_2
        self.match_point_3 = match_point_3
        self.coverage_ratio = float(coverage_ratio)
        self.min_patch_size = int(min_patch_size)
        self.model_workers = (
            None if model_workers is None else int(model_workers)
        )
        self.model_memory_fraction = float(model_memory_fraction)

    def forward(self) -> None:
        run(
            x_load_path=self.x_load_path,
            y_load_path=self.y_load_path,
            model_load_fold=self.model_load_fold,
            save_fold=self.save_fold,
            downsample_factor=self.downsample_factor,
            chunk_size=self.chunk_size,
            seam_width=self.seam_width,
            tile_size=self.tile_size,
            match_point_1=self.match_point_1,
            match_point_2=self.match_point_2,
            match_point_3=self.match_point_3,
            coverage_ratio=self.coverage_ratio,
            min_patch_size=self.min_patch_size,
            model_workers=self.model_workers,
            model_memory_fraction=self.model_memory_fraction,
        )
