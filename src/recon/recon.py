import numpy as np
import tifffile
import time
import tqdm
import zarr
from collections.abc import Mapping, Sequence
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any

from .decon import Decon
from .distort import Distort
from .stitch import Stitch


class ProcBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs) -> None:
        self.proc_text = "proc=00.00s/ch"
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self) -> dict[str, object]:
        d = super().format_dict
        total = d.get("total")
        if total:
            width = len(str(int(total)))
            d["n_fmt"] = f"{int(d.get('n', 0)):0{width}d}"
            d["total_fmt"] = f"{int(total):0{width}d}"
        d["proc"] = self.proc_text
        return d


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain(v) for v in value]
    return value


def _to_plain_dict(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return {str(k): _to_plain(v) for k, v in value.items()}


def run(
    x_load_path: str,
    y_save_path: str,
    frame: Sequence[int],
    chunk_size: int,
    queue_size: int,
    dtype: str,
    distortion: Mapping[str, Any],
    decon: Mapping[str, Any],
    stitch: Mapping[str, Any],
) -> None:
    distortion_cfg = _to_plain_dict(distortion, "distortion")
    decon_cfg = _to_plain_dict(decon, "decon")
    stitch_cfg = _to_plain_dict(stitch, "stitch")

    stages: list[tuple[str, object]] = []
    if bool(distortion_cfg.get("enable", False)):
        distortion_stage = Distort(**distortion_cfg, dtype=dtype)
        stages.append(("distort", distortion_stage.forward))
    if bool(decon_cfg.get("enable", False)):
        decon_stage = Decon(**decon_cfg, dtype=dtype)
        stages.append(("deconvolve", decon_stage.forward))
    if bool(stitch_cfg.get("enable", False)):
        stitch_stage = Stitch(**stitch_cfg, dtype=dtype)
        stages.append(("stitch", stitch_stage.forward))

    queue_size = max(1, int(queue_size))
    frame_cfg = list(frame)
    if len(frame_cfg) != 2:
        raise ValueError("frame must be [start, end], with end=-1 meaning full range.")
    frame_start = int(frame_cfg[0])
    frame_end_cfg = int(frame_cfg[1])
    sentinel = object()
    stage_bar_format = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{proc}]"
    )

    with tifffile.TiffFile(x_load_path) as tif:
        video = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            video = video[None, ...]
        num_frames = int(video.shape[0])
        if frame_start < 0:
            raise ValueError("frame start must be >= 0.")
        if frame_start >= num_frames:
            raise ValueError(
                f"frame start {frame_start} is out of range for {num_frames} frames."
            )
        frame_end = num_frames if frame_end_cfg == -1 else min(frame_end_cfg, num_frames)
        if frame_end <= frame_start:
            raise ValueError(
                f"Invalid frame range [{frame_start}, {frame_end_cfg}] for {num_frames} frames."
            )

        total_frames = frame_end - frame_start
        total_chunks = (total_frames + chunk_size - 1) // chunk_size
        queues = [Queue(maxsize=queue_size) for _ in range(len(stages) + 1)]
        stop_event = Event()
        errors: Queue = Queue()
        stat_lock = Lock()
        stage_stats: dict[str, dict[str, float]] = {}

        def update_proc_stats(name: str, dt: float, bar: ProcBar) -> None:
            with stat_lock:
                stats = stage_stats.setdefault(name, {"sum": 0.0, "n": 0.0})
                stats["sum"] += float(dt)
                stats["n"] += 1.0
                avg = stats["sum"] / stats["n"] if stats["n"] > 0 else 0.0
            bar.proc_text = f"proc={avg:05.2f}s/ch"

        bars: list[tqdm.tqdm] = []
        recon_bar = ProcBar(
            total=total_chunks,
            unit="chunk",
            desc="reconstruct",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        bars.append(recon_bar)
        read_bar = ProcBar(
            total=total_chunks,
            unit="chunk",
            desc="- read",
            position=1,
            leave=True,
            dynamic_ncols=True,
            bar_format=stage_bar_format,
        )
        bars.append(read_bar)
        stage_bars: dict[str, ProcBar] = {}
        for i, (stage_name, _) in enumerate(stages, start=2):
            stage_bar = ProcBar(
                total=total_chunks,
                unit="chunk",
                desc=f"- {stage_name}",
                position=i,
                leave=True,
                dynamic_ncols=True,
                bar_format=stage_bar_format,
            )
            stage_bars[stage_name] = stage_bar
            bars.append(stage_bar)
        write_bar = ProcBar(
            total=total_chunks,
            unit="chunk",
            desc="- write",
            position=len(stages) + 2,
            leave=True,
            dynamic_ncols=True,
            bar_format=stage_bar_format,
        )
        bars.append(write_bar)

        def record_error(exc: Exception) -> None:
            if errors.empty():
                errors.put(exc)
            stop_event.set()

        def safe_put(q: Queue, item: object) -> None:
            while True:
                try:
                    q.put(item, timeout=0.2)
                    return
                except Full:
                    if stop_event.is_set():
                        return

        def reader() -> None:
            try:
                for chunk_idx, start in enumerate(range(frame_start, frame_end, chunk_size)):
                    if stop_event.is_set():
                        break
                    end = min(start + chunk_size, frame_end)
                    t0 = time.perf_counter()
                    chunk = np.asarray(video[start:end], dtype=dtype)
                    dt = time.perf_counter() - t0
                    safe_put(queues[0], (chunk_idx, chunk))
                    read_bar.update(1)
                    update_proc_stats("read", dt, read_bar)
            except Exception as exc:
                record_error(RuntimeError(f"reader failed: {exc}"))
            finally:
                safe_put(queues[0], sentinel)

        def stage_worker(
            stage_name: str,
            stage_fn: object,
            in_queue: Queue,
            out_queue: Queue,
        ) -> None:
            try:
                while True:
                    item = in_queue.get()
                    if item is sentinel:
                        safe_put(out_queue, sentinel)
                        break
                    if stop_event.is_set():
                        continue
                    chunk_idx, chunk = item
                    t0 = time.perf_counter()
                    out = stage_fn(chunk)
                    dt = time.perf_counter() - t0
                    safe_put(out_queue, (chunk_idx, out))
                    stage_bars[stage_name].update(1)
                    update_proc_stats(stage_name, dt, stage_bars[stage_name])
            except Exception as exc:
                record_error(RuntimeError(f"{stage_name} failed: {exc}"))
                safe_put(out_queue, sentinel)

        threads: list[Thread] = []
        t_read = Thread(target=reader, name="pipeline-reader", daemon=True)
        t_read.start()
        threads.append(t_read)

        for i, (stage_name, stage_fn) in enumerate(stages):
            t = Thread(
                target=stage_worker,
                args=(stage_name, stage_fn, queues[i], queues[i + 1]),
                name=f"pipeline-{stage_name}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        pending: dict[int, np.ndarray] = {}
        next_chunk_idx = 0
        pipeline_done = False
        try:
            with tifffile.TiffWriter(y_save_path, bigtiff=True) as writer:
                while not pipeline_done:
                    if not errors.empty():
                        raise errors.get()
                    try:
                        item = queues[-1].get(timeout=0.2)
                    except Empty:
                        continue

                    if item is sentinel:
                        pipeline_done = True
                        continue

                    chunk_idx, chunk = item
                    pending[chunk_idx] = np.asarray(chunk, dtype=dtype)

                    while next_chunk_idx in pending:
                        chunk_out = pending.pop(next_chunk_idx)
                        t0 = time.perf_counter()
                        if chunk_out.ndim == 2:
                            chunk_out = chunk_out[None, ...]
                        for frame in chunk_out:
                            writer.write(frame, contiguous=True)
                        dt = time.perf_counter() - t0
                        next_chunk_idx += 1
                        write_bar.update(1)
                        update_proc_stats("write", dt, write_bar)
                        recon_bar.update(1)

                while next_chunk_idx in pending:
                    chunk_out = pending.pop(next_chunk_idx)
                    t0 = time.perf_counter()
                    if chunk_out.ndim == 2:
                        chunk_out = chunk_out[None, ...]
                    for frame in chunk_out:
                        writer.write(frame, contiguous=True)
                    dt = time.perf_counter() - t0
                    next_chunk_idx += 1
                    write_bar.update(1)
                    update_proc_stats("write", dt, write_bar)
                    recon_bar.update(1)
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=1.0)
            for bar in bars:
                bar.close()

        if not errors.empty():
            raise errors.get()
        if next_chunk_idx != total_chunks:
            raise RuntimeError(
                f"Pipeline finished early: wrote {next_chunk_idx}/{total_chunks} chunks."
            )


class Recon:
    def __init__(
        self,
        x_load_path: str,
        y_save_path: str,
        frame: Sequence[int],
        chunk_size: int,
        queue_size: int,
        dtype: str,
        distortion: Mapping[str, Any],
        decon: Mapping[str, Any],
        stitch: Mapping[str, Any],
        **kwargs,
    ) -> None:
        self.x_load_path = x_load_path
        self.y_save_path = y_save_path
        self.frame = [int(frame[0]), int(frame[1])]
        self.chunk_size = int(chunk_size)
        self.queue_size = int(queue_size)
        self.dtype = str(dtype)
        self.distortion = _to_plain_dict(distortion, "distortion")
        self.decon = _to_plain_dict(decon, "decon")
        self.stitch = _to_plain_dict(stitch, "stitch")

    def forward(self) -> None:
        run(
            x_load_path=self.x_load_path,
            y_save_path=self.y_save_path,
            frame=self.frame,
            chunk_size=self.chunk_size,
            queue_size=self.queue_size,
            dtype=self.dtype,
            distortion=self.distortion,
            decon=self.decon,
            stitch=self.stitch,
        )
