import numpy as np
import os
import tifffile
import tqdm
import zarr


def run(
    y_load_path: str,
    y_save_path: str,
    chunk_size: int,
    dtype: str,
) -> None:
    with tifffile.TiffFile(y_load_path) as tif:
        video: zarr.Array = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            video = video[None, ...]
        total_chunks = (video.shape[0] + chunk_size - 1) // chunk_size
        global_min = float("inf")
        global_max = float("-inf")
        with tqdm.tqdm(
            total=total_chunks,
            unit="chunk",
            desc="normalize(1/2)",
            dynamic_ncols=True,
        ) as bar:
            for start in range(0, video.shape[0], chunk_size):
                end = min(start + chunk_size, video.shape[0])
                chunk = np.asarray(video[start:end], dtype=dtype)
                if chunk.ndim == 2:
                    chunk = chunk[None, ...]
                global_min = min(global_min, float(np.min(chunk)))
                global_max = max(global_max, float(np.max(chunk)))
                bar.update(1)

    same_path = (
        os.path.abspath(y_load_path) == os.path.abspath(y_save_path)
    )
    if not same_path:
        tmp_path = y_save_path
    else:
        tmp_path = y_save_path + ".tmp_norm.tif"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with tifffile.TiffFile(y_load_path) as tif:
        video: zarr.Array = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2:
            video = video[None, ...]
        total_chunks = (video.shape[0] + chunk_size - 1) // chunk_size
        with tqdm.tqdm(
            total=total_chunks,
            unit="chunk",
            desc="normalize(2/2)",
            dynamic_ncols=True,
        ) as bar:
            with tifffile.TiffWriter(tmp_path, bigtiff=True) as writer:
                for start in range(0, video.shape[0], chunk_size):
                    end = min(start + chunk_size, video.shape[0])
                    chunk = np.asarray(video[start:end], dtype=dtype)
                    if chunk.ndim == 2:
                        chunk = chunk[None, ...]
                    if global_max > global_min:
                        chunk = (
                            (chunk - global_min) /
                            (global_max - global_min)
                        ).astype(dtype, copy=False)
                    else:
                        chunk = np.zeros_like(chunk, dtype=dtype)
                    for frame in chunk:
                        writer.write(frame, contiguous=True)
                    bar.update(1)

    if same_path:
        os.replace(tmp_path, y_save_path)


class Normalize:
    def __init__(
        self,
        y_load_path: str,
        y_save_path: str,
        chunk_size: int,
        dtype: str,
        **kwargs,
    ) -> None:
        self.y_load_path = y_load_path
        self.y_save_path = y_save_path
        self.chunk_size = int(chunk_size)
        self.dtype = str(dtype)

    def forward(self) -> None:
        run(
            y_load_path=self.y_load_path,
            y_save_path=self.y_save_path,
            chunk_size=self.chunk_size,
            dtype=self.dtype,
        )
