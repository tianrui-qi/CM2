import hydra
import numpy as np
import omegaconf
import os
import tifffile
import tqdm
import zarr


@hydra.main(
    version_base=None, config_path="../config", config_name="normalize"
)
def main(cfg: omegaconf.DictConfig) -> None:
    with tifffile.TiffFile(cfg.load_path) as tif:
        video: zarr.Array = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2: video = video[None, ...]
        total_chunks = (
            (video.shape[0] + cfg.chunk_size - 1) // cfg.chunk_size
        )
        global_min = float("inf")
        global_max = float("-inf")
        with tqdm.tqdm(
            total=total_chunks, unit="chunk", desc="normalize(1/2)",
            dynamic_ncols=True,
        ) as bar:
            for start in range(0, video.shape[0], cfg.chunk_size):
                end = min(start + cfg.chunk_size, video.shape[0])
                chunk = np.asarray(video[start:end], dtype=cfg.dtype)
                if chunk.ndim == 2:
                    chunk = chunk[None, ...]
                global_min = min(global_min, float(np.min(chunk)))
                global_max = max(global_max, float(np.max(chunk)))
                bar.update(1)

    same_path = (
        os.path.abspath(cfg.load_path) == os.path.abspath(cfg.save_path)
    )
    if not same_path: tmp_path = cfg.save_path
    else: tmp_path = cfg.save_path + ".tmp_norm.tif"
    if os.path.exists(tmp_path): os.remove(tmp_path)

    with tifffile.TiffFile(cfg.load_path) as tif:
        video: zarr.Array = zarr.open(tif.aszarr(), mode="r")
        if video.ndim == 2: video = video[None, ...]
        total_chunks = (
            (video.shape[0] + cfg.chunk_size - 1) // cfg.chunk_size
        )
        with tqdm.tqdm(
            total=total_chunks, unit="chunk", desc="normalize(2/2)",
            dynamic_ncols=True,
        ) as bar:
            with tifffile.TiffWriter(tmp_path, bigtiff=True) as writer:
                for start in range(0, video.shape[0], cfg.chunk_size):
                    end = min(start + cfg.chunk_size, video.shape[0])
                    chunk = np.asarray(video[start:end], dtype=cfg.dtype)
                    if chunk.ndim == 2:
                        chunk = chunk[None, ...]
                    if global_max > global_min:
                        chunk = (
                            (chunk - global_min) / 
                            (global_max - global_min)
                        ).astype(cfg.dtype, copy=False)
                    else:
                        chunk = np.zeros_like(chunk, dtype=cfg.dtype)
                    for frame in chunk: 
                        writer.write(frame, contiguous=True)
                    bar.update(1)

    if same_path: os.replace(tmp_path, cfg.save_path)


if __name__ == "__main__": main()