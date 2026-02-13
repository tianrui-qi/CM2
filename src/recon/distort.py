import numpy as np
import tifffile
from scipy.io import loadmat
from scipy.ndimage import map_coordinates


class Distort:
    def __init__(
        self,
        flatfield_path: str,
        prepared_data_path: str,
        xd_key: str = "xd_img_sub",
        yd_key: str = "yd_img_sub",
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.flatfield = np.asarray(tifffile.imread(flatfield_path), dtype=self.dtype)
        self.xd_sub, self.yd_sub = self._load_distortion_maps(
            prepared_data_path=prepared_data_path,
            xd_key=xd_key,
            yd_key=yd_key,
        )

        if self.flatfield.ndim != 2:
            raise ValueError("flatfield must be a 2D array.")

    def forward(self, frames: np.ndarray) -> np.ndarray:
        chunk, is_single = self._to_chunk(frames)
        if chunk.shape[1:] != self.flatfield.shape:
            raise ValueError(
                "Raw frame shape and flatfield shape must match. "
                f"Got raw={chunk.shape[1:]}, flatfield={self.flatfield.shape}."
            )

        corrected = np.stack(
            [self._correct_frame(chunk[i]) for i in range(chunk.shape[0])],
            axis=0,
        ).astype(self.dtype, copy=False)
        return corrected[0] if is_single else corrected

    @staticmethod
    def _to_chunk(frames: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(frames)
        if arr.ndim == 2:
            return arr[None, ...], True
        if arr.ndim == 3:
            return arr, False
        raise ValueError("Input must be a 2D frame or 3D chunk (T, H, W).")

    def _load_distortion_maps(
        self,
        prepared_data_path: str,
        xd_key: str,
        yd_key: str,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        data = loadmat(prepared_data_path)
        if xd_key not in data or yd_key not in data:
            raise KeyError(
                f"Cannot find keys '{xd_key}' and '{yd_key}' in {prepared_data_path}."
            )

        xd_list = [np.asarray(v, dtype=self.dtype) for v in data[xd_key].ravel()]
        yd_list = [np.asarray(v, dtype=self.dtype) for v in data[yd_key].ravel()]
        if len(xd_list) != 4 or len(yd_list) != 4:
            raise ValueError("Expected 4 sub distortion maps in both xd and yd cell arrays.")

        return (
            (xd_list[0], xd_list[1], xd_list[2], xd_list[3]),
            (yd_list[0], yd_list[1], yd_list[2], yd_list[3]),
        )

    def _correct_frame(self, frame: np.ndarray) -> np.ndarray:
        img_raw = np.asarray(frame, dtype=self.dtype)
        eps = np.finfo(self.dtype).eps
        img_flat = img_raw / np.maximum(self.flatfield, eps)

        h, w = img_flat.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("Input frame height and width must be even.")
        half_h = h // 2
        half_w = w // 2

        img_sub = (
            img_flat[:half_h, half_w:],
            img_flat[:half_h, :half_w],
            img_flat[half_h:, :half_w],
            img_flat[half_h:, half_w:],
        )

        corrected_sub = []
        for i in range(4):
            if img_sub[i].shape != self.xd_sub[i].shape or img_sub[i].shape != self.yd_sub[i].shape:
                raise ValueError(
                    "Distortion map shape mismatch with sub-image shape. "
                    f"sub={img_sub[i].shape}, xd={self.xd_sub[i].shape}, yd={self.yd_sub[i].shape}."
                )
            corrected_sub.append(
                self._interp2_linear_zero_fill(img_sub[i], self.xd_sub[i], self.yd_sub[i])
            )

        corrected = np.empty_like(img_flat, dtype=self.dtype)
        corrected[:half_h, half_w:] = corrected_sub[0]
        corrected[:half_h, :half_w] = corrected_sub[1]
        corrected[half_h:, :half_w] = corrected_sub[2]
        corrected[half_h:, half_w:] = corrected_sub[3]
        return corrected

    def _interp2_linear_zero_fill(
        self,
        img: np.ndarray,
        xq: np.ndarray,
        yq: np.ndarray,
    ) -> np.ndarray:
        # MATLAB interp2(img, xq, yq, "linear", 0):
        # xq/yq are 1-based, scipy uses 0-based (row, col) coordinates.
        coords = np.vstack(
            [
                (np.asarray(yq, dtype=self.dtype) - 1.0).ravel(),
                (np.asarray(xq, dtype=self.dtype) - 1.0).ravel(),
            ]
        )
        sampled = map_coordinates(
            np.asarray(img, dtype=self.dtype),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return sampled.reshape(xq.shape).astype(self.dtype, copy=False)
