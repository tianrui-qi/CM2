from typing import Literal

import numpy as np
import scipy.fft
import scipy.io
import torch


class Decon:
    def __init__(
        self,
        prepared_data_path: str,
        psf_key: str = "psfs_denoised",
        patch_center_key: str = "undistorted_points",
        marker_index_local: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ] = ((13, 10), (11, 10), (11, 8), (13, 8)),
        lambda_value: float = 0.0206913808111479,
        mode: Literal["global", "patch"] = "patch",
        edge_protection: int = 40,
        overlap: int = 40,
        brightness_matching: bool = False,
        enable_gpu: bool = False,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        self.mode = mode
        self.marker_index_local = np.asarray(marker_index_local, dtype=np.int64)
        self.dtype = np.dtype(dtype)
        self.lambda_value = self.dtype.type(lambda_value)
        self.edge_protection = int(edge_protection)
        self.overlap = int(overlap)
        self.brightness_matching = bool(brightness_matching)
        self.enable_gpu = bool(enable_gpu)
        self._gpu_device = None
        if self.enable_gpu:
            self._gpu_device = torch.device("cuda")

        data = scipy.io.loadmat(prepared_data_path)
        self.psfs = np.asarray(data[psf_key], dtype=self.dtype)
        self.patch_centers = np.asarray(data[patch_center_key], dtype=self.dtype)

        self._sub_shape: tuple[int, int] | None = None
        self._sub_plans: list[dict[str, object]] | None = None

    def forward(self, frames: np.ndarray) -> np.ndarray:
        chunk, is_single = self._to_chunk(frames)
        deconv = np.stack(
            [self._decon_frame(f) for f in chunk], axis=0
        ).astype(self.dtype, copy=False)
        return deconv[0] if is_single else deconv

    @staticmethod
    def _to_chunk(frames: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(frames)
        if arr.ndim == 2:
            return arr[None, ...], True
        if arr.ndim == 3:
            return arr, False
        raise ValueError("Input must be a 2D frame or 3D chunk (T, H, W).")

    def _decon_frame(self, frame: np.ndarray) -> np.ndarray:
        img = np.asarray(frame, dtype=self.dtype)
        h, w = img.shape

        half_h = h // 2
        half_w = w // 2
        sub_shape = (half_h, half_w)
        self._ensure_sub_plans(sub_shape)

        sub_imgs = [
            img[:half_h, half_w:],
            img[:half_h, :half_w],
            img[half_h:, :half_w],
            img[half_h:, half_w:],
        ]
        deconv_sub = [
            self._decon_subimage(
                sub_img=sub_imgs[j], 
                sub_plan=self._sub_plans[j]
            ) for j in range(4)
        ]

        out = np.empty_like(img, dtype=self.dtype)
        out[:half_h, half_w:] = deconv_sub[0]
        out[:half_h, :half_w] = deconv_sub[1]
        out[half_h:, :half_w] = deconv_sub[2]
        out[half_h:, half_w:] = deconv_sub[3]
        return out.astype(self.dtype, copy=False)

    def _decon_subimage(
        self, sub_img: np.ndarray, sub_plan: dict[str, object]
    ) -> np.ndarray:
        edge = self.edge_protection
        global_fft = sub_plan["global_fft"]
        if global_fft is not None:
            img_edge = np.pad(sub_img, ((edge, edge), (edge, edge)), mode="edge")
            deconv_edge = self._apply_fft_plan(img_edge, global_fft)
            if edge > 0: 
                return deconv_edge[
                    edge:-edge, edge:-edge
                ].astype(self.dtype, copy=False)
            return deconv_edge.astype(self.dtype, copy=False)

        patch_grid = sub_plan["patch_grid"]
        num_x = len(patch_grid)
        num_y = len(patch_grid[0]) if num_x > 0 else 0
        out = np.zeros_like(sub_img, dtype=self.dtype)

        if not self.brightness_matching:
            for x in range(num_x):
                for y in range(num_y):
                    plan = patch_grid[x][y]
                    slice_y = plan["slice_y"]
                    slice_x = plan["slice_x"]
                    patch = sub_img[slice_y, slice_x]
                    patch_edge = np.pad(
                        patch, ((edge, edge), (edge, edge)), mode="edge"
                    )
                    deconv_edge = self._apply_fft_plan(patch_edge, plan["fft_plan"])
                    if edge > 0:
                        deconv_patch = deconv_edge[edge:-edge, edge:-edge]
                    else:
                        deconv_patch = deconv_edge
                    alpha = plan["alpha"]
                    if alpha is not None:
                        deconv_patch *= alpha
                    out[slice_y, slice_x] += deconv_patch
            return out.astype(self.dtype, copy=False)

        deconv_patches: list[list[np.ndarray]] = [
            [np.empty((0, 0), dtype=self.dtype) for _ in range(num_y)]
            for _ in range(num_x)
        ]
        overlap = self.overlap

        for x in range(num_x):
            for y in range(num_y):
                plan = patch_grid[x][y]
                patch = sub_img[plan["slice_y"], plan["slice_x"]]
                patch_edge = np.pad(
                    patch, ((edge, edge), (edge, edge)), mode="edge"
                )
                deconv_edge = self._apply_fft_plan(patch_edge, plan["fft_plan"])
                if edge > 0:
                    deconv_patch = deconv_edge[edge:-edge, edge:-edge]
                else:
                    deconv_patch = deconv_edge
                deconv_patches[x][y] = deconv_patch

        for x in range(num_x):
            for y in range(num_y):
                plan = patch_grid[x][y]
                deconv_patch_temp = np.array(deconv_patches[x][y], copy=True)
                ratio = 1.0
                if overlap > 0:
                    ratio_x = 1.0
                    ratio_y = 1.0
                    if x > 0:
                        region_a = deconv_patches[x - 1][y][:, -overlap:]
                        region_b = deconv_patch_temp[:, :overlap]
                        denom = float(np.mean(region_b))
                        ratio_x = float(np.mean(region_a) / denom) if denom != 0 else 1.0
                    if y > 0:
                        region_a = deconv_patches[x][y - 1][:overlap, :]
                        region_b = deconv_patch_temp[-overlap:, :]
                        denom = float(np.mean(region_b))
                        ratio_y = float(np.mean(region_a) / denom) if denom != 0 else 1.0
                    ratio = (
                        float(x > 0 and y == 0) * ratio_x
                        + float(x == 0 and y > 0) * ratio_y
                        + float((x > 0 and y > 0) or (x == 0 and y == 0))
                        * np.sqrt(ratio_x * ratio_y)
                    )
                alpha = plan["alpha"]
                if alpha is not None:
                    deconv_patch_temp *= alpha
                out[plan["slice_y"], plan["slice_x"]] += ratio * deconv_patch_temp

        return out.astype(self.dtype, copy=False)

    def _ensure_sub_plans(self, sub_shape: tuple[int, int]) -> None:
        if self._sub_shape == sub_shape and self._sub_plans is not None:
            return
        self._sub_shape = sub_shape
        self._sub_plans = [
            self._build_sub_plan(j, sub_shape) for j in range(4)
        ]

    def _build_sub_plan(
        self, sub_index: int, sub_shape: tuple[int, int]
    ) -> dict[str, object]:
        sub_h, sub_w = sub_shape
        edge = self.edge_protection
        marker_x = int(self.marker_index_local[sub_index, 0]) - 1
        marker_y = int(self.marker_index_local[sub_index, 1]) - 1

        psf_sub = np.asarray(self.psfs[:, :, :, :, sub_index], dtype=self.dtype)
        patch_center_sub = np.asarray(
            self.patch_centers[:, :, :, sub_index], dtype=self.dtype
        ).copy()

        if patch_center_sub[marker_x, marker_y, 0] > 2000:
            patch_center_sub[:, :, 0] -= 2000
        if patch_center_sub[marker_x, marker_y, 1] > 1500:
            patch_center_sub[:, :, 1] -= 1500

        if self.mode == "global":
            kernel = psf_sub[:, :, marker_x, marker_y]
            fft_plan = self._build_fft_plan(
                kernel=kernel,
                img_h=sub_h + 2 * edge,
                img_w=sub_w + 2 * edge,
            )
            return {"global_fft": fft_plan, "patch_grid": None}

        bounds = self._split_patch_bounds(
            img_h=sub_h,
            img_w=sub_w,
            patch_center=patch_center_sub,
            overlap=self.overlap,
        )
        num_x = len(bounds)
        num_y = len(bounds[0]) if num_x > 0 else 0
        patch_grid: list[list[dict[str, object]]] = [
            [{} for _ in range(num_y)] for _ in range(num_x)
        ]

        for x in range(num_x):
            for y in range(num_y):
                x0, x1, y0, y1 = bounds[x][y]
                patch_h = y1 - y0 + 1
                patch_w = x1 - x0 + 1

                overlap_top = self.overlap * int(y != num_y - 1)
                overlap_bottom = self.overlap * int(y != 0)
                overlap_left = self.overlap * int(x != 0)
                overlap_right = self.overlap * int(x != num_x - 1)

                alpha = None
                if self.overlap > 0:
                    alpha = self._alpha_mask(
                        h=patch_h,
                        w=patch_w,
                        overlap_top=overlap_top,
                        overlap_bottom=overlap_bottom,
                        overlap_left=overlap_left,
                        overlap_right=overlap_right,
                    )

                fft_plan = self._build_fft_plan(
                    kernel=np.asarray(psf_sub[:, :, x, y], dtype=self.dtype),
                    img_h=patch_h + 2 * edge,
                    img_w=patch_w + 2 * edge,
                )

                patch_grid[x][y] = {
                    "slice_x": slice(x0, x1 + 1),
                    "slice_y": slice(y0, y1 + 1),
                    "alpha": alpha,
                    "fft_plan": fft_plan,
                }

        return {"global_fft": None, "patch_grid": patch_grid}

    def _build_fft_plan(
        self, kernel: np.ndarray, img_h: int, img_w: int
    ) -> dict[str, object]:
        k_h, k_w = kernel.shape
        fft_h = img_h + k_h - 1
        fft_w = img_w + k_w - 1

        start_y = int(np.ceil((k_h - 1) / 2.0) + 1)
        start_x = int(np.ceil((k_w - 1) / 2.0) + 1)
        pad_top = start_y - 1
        pad_left = start_x - 1
        pad_bottom = fft_h - img_h - pad_top
        pad_right = fft_w - img_w - pad_left

        f_kernel = scipy.fft.rfft2(
            np.asarray(kernel, dtype=self.dtype),
            s=(fft_h, fft_w),
            workers=-1,
        )
        filt = np.conj(f_kernel) / (np.abs(f_kernel) ** 2 + self.lambda_value)
        if self.dtype == np.float32:
            filt = filt.astype(np.complex64, copy=False)

        return {
            "img_h": img_h,
            "img_w": img_w,
            "fft_h": fft_h,
            "fft_w": fft_w,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "filt": filt,
            "filt_torch": None,
        }

    def _apply_fft_plan(
        self, img: np.ndarray, plan: dict[str, object]
    ) -> np.ndarray:
        img_h = int(plan["img_h"])
        img_w = int(plan["img_w"])
        fft_h = int(plan["fft_h"])
        fft_w = int(plan["fft_w"])
        pad_top = int(plan["pad_top"])
        pad_bottom = int(plan["pad_bottom"])
        pad_left = int(plan["pad_left"])
        pad_right = int(plan["pad_right"])
        filt = plan["filt"]

        if self.enable_gpu:
            filt_torch = plan["filt_torch"]
            if filt_torch is None:
                filt_torch = torch.as_tensor(filt, device=self._gpu_device)
                plan["filt_torch"] = filt_torch
            img_t = torch.as_tensor(
                np.asarray(img, dtype=self.dtype),
                device=self._gpu_device,
            )
            img_padded = torch.nn.functional.pad(
                img_t,
                (pad_left, pad_right, pad_top, pad_bottom),
            )
            f_img = torch.fft.rfft2(img_padded, s=(fft_h, fft_w))
            deconv = torch.fft.irfft2(f_img * filt_torch, s=(fft_h, fft_w))
            deconv = deconv[:img_h, :img_w]
            deconv = torch.clamp(deconv, min=0)
            deconv = (
                deconv.detach().cpu().numpy().astype(self.dtype, copy=False)
            )
            return deconv

        img_padded = np.pad(
            np.asarray(img, dtype=self.dtype),
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )
        f_img = scipy.fft.rfft2(img_padded, s=(fft_h, fft_w), workers=-1)
        deconv = scipy.fft.irfft2(f_img * filt, s=(fft_h, fft_w), workers=-1)
        deconv = deconv[:img_h, :img_w]
        np.maximum(deconv, 0, out=deconv)
        return deconv.astype(self.dtype, copy=False)

    def _split_patch_bounds(
        self,
        img_h: int,
        img_w: int,
        patch_center: np.ndarray,
        overlap: int,
    ) -> list[list[tuple[int, int, int, int]]]:
        num_x, num_y = patch_center.shape[:2]
        overlap_half = overlap / 2.0
        bounds: list[list[tuple[int, int, int, int]]] = [
            [(-1, -1, -1, -1) for _ in range(num_y)] for _ in range(num_x)
        ]

        for x in range(num_x):
            for y in range(num_y):
                if x == 0:
                    x_left = 1.0
                else:
                    x_left = (
                        self._matlab_round(
                            (patch_center[x, y, 0] + patch_center[x - 1, y, 0]) / 2.0
                        )
                        - overlap_half
                    )
                if x == num_x - 1:
                    x_right = float(img_w)
                else:
                    x_right = (
                        self._matlab_round(
                            (patch_center[x, y, 0] + patch_center[x + 1, y, 0]) / 2.0
                            - 1.0
                        )
                        + overlap_half
                    )

                if y == 0:
                    y_bottom = float(img_h)
                else:
                    y_bottom = (
                        self._matlab_round(
                            (patch_center[x, y, 1] + patch_center[x, y - 1, 1]) / 2.0
                            - 1.0
                        )
                        + overlap_half
                    )
                if y == num_y - 1:
                    y_top = 1.0
                else:
                    y_top = (
                        self._matlab_round(
                            (patch_center[x, y, 1] + patch_center[x, y + 1, 1]) / 2.0
                        )
                        - overlap_half
                    )

                x_left_i = int(np.clip(self._matlab_round(x_left), 1, img_w))
                x_right_i = int(np.clip(self._matlab_round(x_right), 1, img_w))
                y_top_i = int(np.clip(self._matlab_round(y_top), 1, img_h))
                y_bottom_i = int(np.clip(self._matlab_round(y_bottom), 1, img_h))

                bounds[x][y] = (
                    x_left_i - 1,
                    x_right_i - 1,
                    y_top_i - 1,
                    y_bottom_i - 1,
                )

        return bounds

    def _alpha_mask(
        self,
        h: int,
        w: int,
        overlap_top: int,
        overlap_bottom: int,
        overlap_left: int,
        overlap_right: int,
    ) -> np.ndarray:
        overlap_top = max(0, min(overlap_top, h))
        overlap_bottom = max(0, min(overlap_bottom, h))
        overlap_left = max(0, min(overlap_left, w))
        overlap_right = max(0, min(overlap_right, w))

        alpha = np.ones((h, w), dtype=self.dtype)
        if overlap_top > 0:
            alpha[:overlap_top, :] = np.repeat(
                self._robust_linspace(0.0, 1.0, overlap_top)[:, None], 
                w, axis=1
            )
        if overlap_bottom > 0:
            alpha[h - overlap_bottom :, :] = np.repeat(
                self._robust_linspace(1.0, 0.0, overlap_bottom)[:, None], 
                w, axis=1
            )
        if overlap_left > 0:
            alpha[:, :overlap_left] *= np.repeat(
                self._robust_linspace(0.0, 1.0, overlap_left)[None, :], 
                h, axis=0
            )
        if overlap_right > 0:
            alpha[:, w - overlap_right :] *= np.repeat(
                self._robust_linspace(1.0, 0.0, overlap_right)[None, :], 
                h, axis=0
            )
        return alpha.astype(self.dtype, copy=False)

    @staticmethod
    def _matlab_round(x: float) -> int:
        if x >= 0:
            return int(np.floor(x + 0.5))
        return int(np.ceil(x - 0.5))

    def _robust_linspace(self, a: float, b: float, n: int) -> np.ndarray:
        if n == 1:
            return np.array([0.5], dtype=self.dtype)
        if n < 1:
            return np.empty((0,), dtype=self.dtype)
        return np.linspace(a, b, n, dtype=self.dtype)
