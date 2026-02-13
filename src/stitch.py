import numpy as np


class Stitch:
    def __init__(
        self,
        match_point_1: (tuple[tuple[int, int], tuple[int, int]]),
        match_point_2: (tuple[tuple[int, int], tuple[int, int]]),
        match_point_3: (tuple[tuple[int, int], tuple[int, int]]),
        match_brightness: bool = False,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        self.match_point_1 = np.asarray(match_point_1, dtype=np.int64)
        self.match_point_2 = np.asarray(match_point_2, dtype=np.int64)
        self.match_point_3 = np.asarray(match_point_3, dtype=np.int64)
        self.match_brightness = match_brightness
        self.dtype = np.dtype(dtype)

    def forward(self, frames: np.ndarray) -> np.ndarray:
        chunk, is_single = self._to_chunk(frames)
        stitched = np.stack(
            [self._stitch_frame(chunk[i]) for i in range(chunk.shape[0])],
            axis=0,
        ).astype(self.dtype, copy=False)

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
        stitch_1 = self._manual_stitch_horizontal(
            np.rot90(img_raw_sub_2, 2),
            np.rot90(img_raw_sub_1, 2),
            self.match_point_1,
        )
        stitch_2 = self._manual_stitch_horizontal(
            np.rot90(img_raw_sub_3, 2),
            np.rot90(img_raw_sub_4, 2),
            self.match_point_2,
        )
        return self._manual_stitch_vertical(
            stitch_1, 
            stitch_2,
            self.match_point_3,
        )

    def _manual_stitch_horizontal(
        self, 
        img_a: np.ndarray, img_b: np.ndarray, match_point: np.ndarray,      
    ) -> np.ndarray:
        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        shift_x = int(w_a - match_point[0, 0] + match_point[1, 0])
        shift_y = int(match_point[1, 1] - match_point[0, 1])
        img_a_overlap_x = self._matlab_range(w_a - shift_x + 1, w_a)
        img_b_overlap_x = self._matlab_range(1, shift_x)
        img_a_overlap_y = self._matlab_range(
            max(1 - shift_y, 1), min(h_b - shift_y, h_a)
        )
        img_b_overlap_y = self._matlab_range(
            max(1 + shift_y, 1), min(h_a + shift_y, h_b)
        )
        stitched = np.zeros(
            (img_a_overlap_y.size, w_a + w_b - img_a_overlap_x.size), 
            dtype=self.dtype,
        )
        stitched[:, :w_a] = self._alpha_blending(
            img_a[img_a_overlap_y, :], 0, 0, 0, shift_x
        )
        if self.match_brightness:
            denom = np.mean(
                img_b[np.ix_(img_b_overlap_y, img_b_overlap_x)]
            )
            ratio = np.mean(
                img_a[np.ix_(img_a_overlap_y, img_a_overlap_x)]
            ) / denom if denom != 0 else 1.0
        else:
            ratio = 1.0
        stitched[:, w_a - shift_x :] += ratio * self._alpha_blending(
            img_b[img_b_overlap_y, :], 0, 0, shift_x, 0
        )
        return stitched

    def _manual_stitch_vertical(
        self,
        img_a: np.ndarray, img_b: np.ndarray, match_point: np.ndarray,
    ) -> np.ndarray:
        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        shift_x = int(match_point[1, 0] - match_point[0, 0])
        shift_y = int(h_a - match_point[0, 1] + match_point[1, 1])
        img_a_overlap_x = self._matlab_range(
            max(1 - shift_x, 1), min(w_b - shift_x, w_a)
        )
        img_b_overlap_x = self._matlab_range(
            max(1 + shift_x, 1), min(w_a + shift_x, w_b)
        )
        img_a_overlap_y = self._matlab_range(h_a - shift_y + 1, h_a)
        img_b_overlap_y = self._matlab_range(1, shift_y)
        stitched = np.zeros(
            (h_a + h_b - img_a_overlap_y.size, img_a_overlap_x.size), 
            dtype=self.dtype
        )
        stitched[:h_a, :] = self._alpha_blending(
            img_a[:, img_a_overlap_x], 0, shift_y, 0, 0
        )
        if self.match_brightness:
            denom = np.mean(
                img_b[np.ix_(img_b_overlap_y, img_b_overlap_x)]
            )
            ratio = np.mean(
                img_a[np.ix_(img_a_overlap_y, img_a_overlap_x)]
            ) / denom if denom != 0 else 1.0
        else:
            ratio = 1.0
        stitched[h_a - shift_y :, :] += ratio * self._alpha_blending(
            img_b[:, img_b_overlap_x], shift_y, 0, 0, 0
        )
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

    def _alpha_blending(
        self, img: np.ndarray,
        overlap_top: int, overlap_bottom: int,
        overlap_left: int, overlap_right: int,
    ) -> np.ndarray:
        m, n = img.shape
        alpha = np.ones((m, n), dtype=self.dtype)
        if overlap_top > 0: alpha[:overlap_top, :] = np.repeat(
            self._robust_linspace(0, 1, overlap_top)[:, None], n, axis=1
        )
        if overlap_bottom > 0: alpha[m - overlap_bottom :, :] = np.repeat(
            self._robust_linspace(1, 0, overlap_bottom)[:, None], n, axis=1
        )
        if overlap_left > 0: alpha[:, :overlap_left] *= np.repeat(
            self._robust_linspace(0, 1, overlap_left)[None, :], m, axis=0
        )
        if overlap_right > 0: alpha[:, n - overlap_right :] *= np.repeat(
            self._robust_linspace(1, 0, overlap_right)[None, :], m, axis=0
        )
        return img * alpha

    def _robust_linspace(self, a: float, b: float, n: int) -> np.ndarray:
        if n == 1:  return np.array([0.5], dtype=self.dtype)
        elif n < 1: return np.empty((0,), dtype=self.dtype)
        else:       return np.linspace(a, b, n, dtype=self.dtype)
