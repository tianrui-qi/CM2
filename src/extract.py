import csv
import json
import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import numpy as np
import omegaconf
import tqdm
import tifffile

import caiman
import caiman.cluster
from caiman.components_evaluation import (
    estimate_components_quality_auto,
    find_activity_intervals,
)
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf import params as params
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import norm as scipy_norm


CNMFE_GROUPS = (
    "data",
    "init",
    "motion",
    "preprocess",
    "temporal",
    "spatial",
    "patch",
    "merging",
)

KNOWN_CNMFE_WARNING_FILTERS = (
    (
        r"divide by zero encountered in remainder",
        r"scipy\.sparse\._dia",
        "scipy.sparse._dia",
    ),
    (
        r"divide by zero encountered in divide",
        r"caiman\.source_extraction\.cnmf\.merging",
        "caiman.source_extraction.cnmf.merging",
    ),
    (
        r"invalid value encountered in divide",
        r"caiman\.source_extraction\.cnmf\.merging",
        "caiman.source_extraction.cnmf.merging",
    ),
)

QUALITY_RENDER_SCALE_FACTOR = 4
QUALITY_RENDER_DPI = 400
QUALITY_DOT_SIZE = 9.0
QUALITY_DOT_ALPHA = 1.0
QUALITY_LABEL_FONT_SIZE = 2.2
QUALITY_LABEL_OFFSET_X = 2.0
QUALITY_LABEL_OFFSET_Y = -1.0
QUALITY_R_VALUE_CMAP = "RdBu"
QUALITY_SNR_CMAP = "RdBu"
QUALITY_R_VALUE_DISPLAY_CENTER_METHOD = "median"
QUALITY_R_VALUE_DISPLAY_CLIP_PERCENTILE = 99.0
QUALITY_SNR_DISPLAY_CENTER_METHOD = "median_log10"
QUALITY_SNR_DISPLAY_CLIP_PERCENTILE = 99.0
QUALITY_SPATIAL_SIGMA_FLOOR_PX = 24.0
QUALITY_SPATIAL_SIGMA_FACTOR = 2.0
QUALITY_GAUSSIAN_DISPLAY_SIGMA_RANGE = 3.0
QUALITY_BOOLEAN_BLUE_RED_CMAP = mcolors.ListedColormap(
    np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
)


@dataclass(frozen=True)
class QualityMetricSpec:
    key: str
    slug: str
    scale: str
    cmap: Any


@dataclass
class QualityRenderComponent:
    patch_name: str
    component_index: int
    x: int
    y: int
    outline_y: np.ndarray
    outline_x: np.ndarray
    metrics: dict[str, float | bool]


QUALITY_METRIC_SPECS: tuple[QualityMetricSpec, ...] = (
    QualityMetricSpec("r_value", "rvalue", "linear", "RdBu"),
    QualityMetricSpec("snr", "snr", "log", "RdBu"),
    QualityMetricSpec("bl", "bl", "linear", "RdBu"),
    QualityMetricSpec("lam", "lam", "log", "RdBu"),
    QualityMetricSpec("neurons_sn", "sn", "log", "RdBu"),
    QualityMetricSpec("g_0", "g0", "linear", "RdBu"),
    QualityMetricSpec("g_1", "g1", "linear", "RdBu"),
    QualityMetricSpec(
        "r_value_unreliable_joint_only",
        "join",
        "boolean",
        QUALITY_BOOLEAN_BLUE_RED_CMAP,
    ),
)
QUALITY_PROFILE_KEYS = tuple(spec.key for spec in QUALITY_METRIC_SPECS)
QUALITY_THRESHOLD_Z_VALUES = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.1), 1)
QUALITY_THRESHOLD_ACCEPT_RGB = np.array([0, 0, 255], dtype=np.uint8)
QUALITY_THRESHOLD_REJECT_RGB = np.array([255, 0, 0], dtype=np.uint8)
QUALITY_THRESHOLD_JOIN_ACCEPT_RGB = np.array([0, 0, 255], dtype=np.uint8)
QUALITY_THRESHOLD_BAD_HIGH_KEYS = frozenset({"bl", "g_1"})


def _to_container(value: Any) -> Any:
    if omegaconf.OmegaConf.is_config(value):
        return omegaconf.OmegaConf.to_container(value, resolve=False)
    return value


@contextmanager
def silence_stdio():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("caiman")
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())
    logging.getLogger().setLevel(logging.ERROR)
    return logger


def configure_warning_filters() -> None:
    for message, module_regex, _ in KNOWN_CNMFE_WARNING_FILTERS:
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=RuntimeWarning,
            module=module_regex,
        )

    existing = os.environ.get("PYTHONWARNINGS", "")
    env_specs = [
        f"ignore:{message}:RuntimeWarning:{module_name}"
        for message, _, module_name in KNOWN_CNMFE_WARNING_FILTERS
    ]
    missing_specs = [spec for spec in env_specs if spec not in existing.split(",")]
    if not missing_specs:
        return
    os.environ["PYTHONWARNINGS"] = ",".join(
        [part for part in [existing, *missing_specs] if part]
    )


def resolve_input_movies(y_load_fold_cfg: Any) -> tuple[list[Path], Path]:
    if omegaconf.OmegaConf.is_config(y_load_fold_cfg):
        raw = omegaconf.OmegaConf.to_container(y_load_fold_cfg, resolve=True)
    else:
        raw = y_load_fold_cfg
    if not isinstance(raw, str):
        raise TypeError(
            f"extract.y_load_fold must be a single path string, got {type(raw).__name__}"
        )

    input_path = Path(raw)
    if input_path.is_file():
        movie_path = input_path.resolve()
        if movie_path.suffix.lower() != ".tif":
            raise ValueError(f"Single input file must end with .tif: {movie_path}")
        return [movie_path], movie_path.parent / "extract"

    if input_path.is_dir():
        movie_paths = sorted(
            [
                p.resolve()
                for p in input_path.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".tif"
                and not p.name.startswith("._")
            ],
            key=lambda p: p.name.lower(),
        )
        if not movie_paths:
            raise ValueError(f"No .tif files found in input directory: {input_path}")
        return movie_paths, input_path.resolve().parent / "extract"

    raise FileNotFoundError(f"Input path does not exist: {raw}")


def is_directory_input(y_load_fold_cfg: Any) -> bool:
    if omegaconf.OmegaConf.is_config(y_load_fold_cfg):
        raw = omegaconf.OmegaConf.to_container(y_load_fold_cfg, resolve=True)
    else:
        raw = y_load_fold_cfg
    if not isinstance(raw, str):
        raise TypeError(
            f"extract.y_load_fold must be a single path string, got {type(raw).__name__}"
        )
    return Path(raw).is_dir()


def resolve_extract_dir(extract_save_fold_cfg: Any, auto_extract_dir: Path) -> Path:
    if extract_save_fold_cfg is None:
        return auto_extract_dir

    if omegaconf.OmegaConf.is_config(extract_save_fold_cfg):
        raw = omegaconf.OmegaConf.to_container(extract_save_fold_cfg, resolve=True)
    else:
        raw = extract_save_fold_cfg
    if not isinstance(raw, str):
        raise TypeError(
            f"extract.extract_save_fold must be a path string or null, got {type(raw).__name__}"
        )
    return Path(raw).resolve()


def build_params_dict(
    cfg: omegaconf.DictConfig,
    movie_path: Path,
) -> dict[str, Any]:
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if not isinstance(cfg_dict, dict):
        raise TypeError("Extract config must be a mapping.")

    payload = {k: cfg_dict[k] for k in CNMFE_GROUPS if k in cfg_dict}
    data_group = dict(payload.get("data", {}))
    data_group["fnames"] = [str(movie_path)]
    payload["data"] = data_group
    motion_group = dict(payload.get("motion", {}))
    motion_group.pop("enable", None)
    payload["motion"] = motion_group
    return payload


def run_single_patch(
    movie_path: Path,
    cfg: omegaconf.DictConfig,
    dview: object,
    n_processes: int,
    extract_dir: Path,
    logger: logging.Logger,
    profile_path: Path | None = None,
) -> None:
    opts = params.CNMFParams(params_dict=build_params_dict(cfg, movie_path))
    bord_px = 0
    memmap_base_name = f"memmap_{movie_path.stem}_{os.getpid()}_{uuid4().hex[:8]}_"

    with silence_stdio():
        if bool(cfg.motion.enable):
            mc = MotionCorrect(
                opts.data["fnames"],
                dview=dview,
                **opts.get_group("motion"),
            )
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if opts.motion["pw_rigid"] else mc.fname_tot_rig
            if opts.motion["pw_rigid"]:
                bord_px = int(
                    np.ceil(
                        np.maximum(
                            np.max(np.abs(mc.x_shifts_els)),
                            np.max(np.abs(mc.y_shifts_els)),
                        )
                    )
                )
            else:
                bord_px = int(np.ceil(np.max(np.abs(mc.shifts_rig))))
            bord_px = 0 if opts.motion["border_nan"] == "copy" else bord_px
            fname_new = caiman.save_memmap(
                fname_mc,
                base_name=memmap_base_name,
                order="C",
                border_to_0=bord_px,
            )
        else:
            fname_new = caiman.save_memmap(
                opts.data["fnames"],
                base_name=memmap_base_name,
                order="C",
                border_to_0=0,
                dview=dview,
            )

        Yr, dims, T = caiman.load_memmap(fname_new)
        images = Yr.T.reshape((T,) + dims, order="F")
    opts.change_params(params_dict={"dims": dims, "border_pix": bord_px})
    logger.info(
        "[%s] min_corr=%s, min_pnr=%s",
        movie_path.name,
        opts.init["min_corr"],
        opts.init["min_pnr"],
    )

    with silence_stdio():
        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
        cnm.fit(images)
    model_path = save_model(cnm, movie_path, extract_dir, logger)
    if profile_path is not None:
        save_neuron_profile_csv(
            cnm=cnm,
            movie_path=movie_path,
            model_path=model_path,
            profile_path=profile_path,
            movie_tyx=np.asarray(images, dtype=np.float32, copy=False),
            logger=logger,
        )


def get_model_dir(extract_dir: Path) -> Path:
    return extract_dir / "model"


def get_profile_dir(extract_dir: Path) -> Path:
    return extract_dir / "profile"


def get_figure_spatialmap_dir(extract_dir: Path) -> Path:
    return extract_dir / "figure-SpatialMap"


def get_figure_pointmap_dir(extract_dir: Path) -> Path:
    return extract_dir / "figure-PointMap"


def get_figure_pointmap_label_dir(extract_dir: Path) -> Path:
    return extract_dir / "figure-PointMap-label"


def get_figure_thresholdstack_dir(extract_dir: Path) -> Path:
    return extract_dir / "figure-ThresholdStack"


def get_model_path(movie_path: Path, extract_dir: Path) -> Path:
    return get_model_dir(extract_dir) / f"{movie_path.name}.hdf5"


def get_profile_csv_path(movie_path: Path, extract_dir: Path) -> Path:
    return get_profile_dir(extract_dir) / f"{movie_path.name}.csv"


def get_stats_path(extract_dir: Path) -> Path:
    return extract_dir / "stats.json"


def get_stats_png_path(extract_dir: Path) -> Path:
    return extract_dir / "stats.png"


def save_model(
    cnm: cnmf.CNMF,
    fname: Path,
    extract_dir: Path,
    logger: logging.Logger,
) -> Path:
    get_model_dir(extract_dir).mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(fname, extract_dir)
    cnm.save(str(model_path))
    logger.info("[%s] model saved: %s", fname.name, model_path)
    return model_path


def load_movie_tyx(movie_path: Path) -> np.ndarray:
    movie_tyx = tifffile.imread(str(movie_path)).astype(np.float32, copy=False)
    if movie_tyx.ndim != 3:
        raise ValueError(
            f"Expected patch movie to be 3D (T, Y, X), got shape={movie_tyx.shape}"
        )
    return movie_tyx


def evaluate_patch_snr_r_values(
    cnm_model: cnmf.CNMF,
    movie_tyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    movie_yxt = np.transpose(movie_tyx, (1, 2, 0))
    quality = dict(cnm_model.params.get_group("quality"))

    _, _, snr_comp, r_values, _ = estimate_components_quality_auto(
        movie_yxt,
        cnm_model.estimates.A,
        cnm_model.estimates.C,
        cnm_model.estimates.b,
        cnm_model.estimates.f,
        cnm_model.estimates.YrA,
        cnm_model.params.get("data", "fr"),
        cnm_model.params.get("data", "decay_time"),
        cnm_model.params.get("init", "gSig"),
        tuple(int(x) for x in movie_yxt.shape[:-1]),
        dview=None,
        min_SNR=float(quality["min_SNR"]),
        r_values_min=float(quality["rval_thr"]),
        use_cnn=False,
        thresh_cnn_min=float(quality["min_cnn_thr"]),
        thresh_cnn_lowest=float(quality["cnn_lowest"]),
        r_values_lowest=float(quality["rval_lowest"]),
        min_SNR_reject=float(quality["SNR_lowest"]),
    )
    return (
        np.asarray(snr_comp, dtype=np.float64),
        np.asarray(r_values, dtype=np.float64),
    )


def estimate_r_value_unreliable_joint_only_flags(
    cnm_model: cnmf.CNMF,
) -> np.ndarray:
    C = np.asarray(cnm_model.estimates.C, dtype=np.float64)
    n_components = int(C.shape[0])
    if n_components == 0:
        return np.zeros(0, dtype=bool)

    A = csc_matrix(cnm_model.estimates.A)
    AA = (A.T * A).toarray()
    nA = np.sqrt(np.asarray(A.power(2).sum(0), dtype=np.float64)).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        AA = AA / np.outer(nA, nA)
    AA = np.nan_to_num(AA, nan=0.0, posinf=0.0, neginf=0.0)
    AA -= np.eye(n_components)

    final_frate = float(cnm_model.params.get("data", "fr"))
    tB = int(np.minimum(-2, np.floor(-5.0 / 30.0 * final_frate)))
    tA = int(np.maximum(5, np.ceil(25.0 / 30.0 * final_frate)))
    loc = find_activity_intervals(C, Npeaks=10, tB=tB, tA=tA, thres=0.3)

    flags = np.zeros(n_components, dtype=bool)
    for component_index in range(n_components):
        if loc[component_index] is None:
            continue
        overlapping_components = np.where(AA[:, component_index] > 0.1)[0]
        indexes = set(loc[component_index])
        for neighbor_index in overlapping_components:
            if loc[neighbor_index] is not None:
                indexes = indexes - set(loc[neighbor_index])
        if len(indexes) == 0:
            flags[component_index] = True
    return flags


def _component_float_vector(value: Any, n_components: int) -> np.ndarray:
    if value is None:
        return np.full(n_components, np.nan, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_components, float(arr), dtype=np.float64)
    arr = np.ravel(arr).astype(np.float64, copy=False)
    if arr.shape[0] != n_components:
        raise ValueError(
            f"Expected component vector of length {n_components}, got shape={arr.shape}"
        )
    return arr


def _component_g_matrix(value: Any, n_components: int) -> np.ndarray:
    if value is None:
        return np.empty((n_components, 0), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n_components, 1), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == n_components:
            return arr.astype(np.float64, copy=False).reshape(n_components, 1)
        return np.broadcast_to(
            arr.astype(np.float64, copy=False).reshape(1, -1),
            (n_components, arr.shape[0]),
        ).copy()
    if arr.ndim == 2:
        if arr.shape[0] != n_components:
            raise ValueError(
                f"Expected g matrix with {n_components} rows, got shape={arr.shape}"
            )
        return arr.astype(np.float64, copy=False)
    raise ValueError(f"Unsupported g shape for component metadata: {arr.shape}")


def save_neuron_profile_csv(
    cnm: cnmf.CNMF,
    movie_path: Path,
    model_path: Path,
    profile_path: Path,
    movie_tyx: np.ndarray,
    logger: logging.Logger,
) -> None:
    estimates = cnm.estimates
    n_components = int(estimates.A.shape[1])
    g_matrix = _component_g_matrix(getattr(estimates, "g", None), n_components)
    g_fieldnames = [f"g_{idx}" for idx in range(int(g_matrix.shape[1]))]
    fieldnames = [
        "component_index",
        "snr",
        "r_value",
        "bl",
        "lam",
        "neurons_sn",
        *g_fieldnames,
        "r_value_unreliable_joint_only",
    ]

    if n_components == 0:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(
            "[%s] neuron profile saved: %s (0 components)",
            movie_path.name,
            profile_path,
        )
        return

    snr_comp, r_values = evaluate_patch_snr_r_values(cnm, movie_tyx)
    r_value_unreliable = estimate_r_value_unreliable_joint_only_flags(cnm)
    if snr_comp.shape[0] != n_components or r_values.shape[0] != n_components:
        raise ValueError(
            f"Metric/component size mismatch for {movie_path.name}: "
            f"components={n_components}, snr={snr_comp.shape[0]}, r={r_values.shape[0]}"
        )
    if r_value_unreliable.shape[0] != n_components:
        raise ValueError(
            f"R-value reliability flag/component size mismatch for {movie_path.name}: "
            f"components={n_components}, flags={r_value_unreliable.shape[0]}"
        )

    bl = _component_float_vector(getattr(estimates, "bl", None), n_components)
    lam = _component_float_vector(getattr(estimates, "lam", None), n_components)
    neurons_sn = _component_float_vector(
        getattr(estimates, "neurons_sn", None), n_components
    )

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for component_index in range(n_components):
            row: dict[str, Any] = {
                "component_index": int(component_index),
                "snr": float(snr_comp[component_index]),
                "r_value": float(r_values[component_index]),
                "bl": float(bl[component_index]),
                "lam": float(lam[component_index]),
                "neurons_sn": float(neurons_sn[component_index]),
                "r_value_unreliable_joint_only": bool(
                    r_value_unreliable[component_index]
                ),
            }
            for g_idx, g_field in enumerate(g_fieldnames):
                row[g_field] = float(g_matrix[component_index, g_idx])
            writer.writerow(row)

    logger.info(
        "[%s] neuron profile saved: %s (%d components)",
        movie_path.name,
        profile_path,
        n_components,
    )


def _load_profile_metric_arrays(extract_dir: Path) -> dict[str, np.ndarray]:
    profile_dir = get_profile_dir(extract_dir)
    rows_by_key: dict[str, list[float | bool]] = {key: [] for key in QUALITY_PROFILE_KEYS}
    for profile_path in sorted(profile_dir.glob("*.csv")):
        if profile_path.name.startswith("._"):
            continue
        with profile_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in QUALITY_PROFILE_KEYS:
                    raw_value = row[key]
                    if key == "r_value_unreliable_joint_only":
                        rows_by_key[key].append(str(raw_value).strip().lower() == "true")
                    else:
                        rows_by_key[key].append(float(raw_value))

    arrays: dict[str, np.ndarray] = {}
    for spec in QUALITY_METRIC_SPECS:
        if spec.scale == "boolean":
            arrays[spec.key] = np.asarray(rows_by_key[spec.key], dtype=bool)
        else:
            arrays[spec.key] = np.asarray(rows_by_key[spec.key], dtype=np.float64)
    return arrays


def write_extract_stats_json(extract_dir: Path, logger: logging.Logger) -> None:
    metric_arrays = _load_profile_metric_arrays(extract_dir)
    payload: dict[str, Any] = {}
    for spec in QUALITY_METRIC_SPECS:
        values = metric_arrays[spec.key]
        if spec.scale == "boolean":
            payload[spec.key] = {
                "false_count": int(np.count_nonzero(~values)),
                "true_count": int(np.count_nonzero(values)),
            }
            continue

        _, fit_mean, fit_std, meta = _fit_metric_distribution(spec, values)
        metric_payload: dict[str, Any] = {
            "scale": str(meta["scale"]),
            "mean": float(fit_mean),
            "std": float(fit_std),
        }
        payload[spec.key] = metric_payload

    stats_path = get_stats_path(extract_dir)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved extract stats to %s", stats_path)


def render_extract_stats_png(extract_dir: Path, logger: logging.Logger) -> None:
    metric_arrays = _load_profile_metric_arrays(extract_dir)
    continuous_specs = tuple(spec for spec in QUALITY_METRIC_SPECS if spec.scale != "boolean")
    titles = {
        "snr": "SNR",
        "r_value": "R Value",
        "bl": "BL",
        "lam": "Lambda",
        "neurons_sn": "Neurons SN",
        "g_0": "g_0",
        "g_1": "g_1",
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, spec in zip(axes[: len(continuous_specs)], continuous_specs, strict=False):
        raw_values = np.asarray(metric_arrays[spec.key], dtype=np.float64)
        raw_values = raw_values[np.isfinite(raw_values)]
        mapped_values, fit_mean, fit_std, meta = _fit_metric_distribution(spec, raw_values)
        metric_norm = mcolors.TwoSlopeNorm(
            vmin=fit_mean - 3.0 * fit_std,
            vcenter=fit_mean,
            vmax=fit_mean + 3.0 * fit_std,
        )
        metric_cmap = _resolve_metric_cmap(spec.cmap)

        _, bin_edges, patches = ax.hist(
            mapped_values,
            bins=80,
            density=True,
            edgecolor="white",
            linewidth=0.6,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for patch, center in zip(patches, bin_centers, strict=False):
            patch.set_facecolor(metric_cmap(metric_norm(center)))

        x_min = min(float(np.min(mapped_values)), fit_mean - 4.0 * fit_std)
        x_max = max(float(np.max(mapped_values)), fit_mean + 4.0 * fit_std)
        xs = np.linspace(x_min, x_max, 1000)
        pdf = scipy_norm.pdf(xs, loc=fit_mean, scale=fit_std)
        ax.plot(xs, pdf, color="black", linewidth=2.0)
        ax.axvline(fit_mean, color="black", linestyle="--", linewidth=1.3)
        for k in (1, 2, 3):
            ax.axvline(fit_mean - k * fit_std, color="black", linestyle="--", linewidth=1.0)
            ax.axvline(fit_mean + k * fit_std, color="black", linestyle="--", linewidth=1.0)

        title_suffix = " (log10 fit)" if str(meta["scale"]) == "log" else " (linear)"
        x_label = f"log10({spec.key})" if str(meta["scale"]) == "log" else spec.key
        ax.set_title(f"{titles[spec.key]}{title_suffix}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")

    for ax in axes[len(continuous_specs):]:
        ax.axis("off")

    out_path = get_stats_png_path(extract_dir)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved extract stats plot to %s", out_path)


def get_quality_render_output_paths(extract_dir: Path) -> dict[str, Path]:
    spatial_dir = get_figure_spatialmap_dir(extract_dir)
    point_dir = get_figure_pointmap_dir(extract_dir)
    point_label_dir = get_figure_pointmap_label_dir(extract_dir)
    threshold_dir = get_figure_thresholdstack_dir(extract_dir)
    paths: dict[str, Path] = {}
    for spec in QUALITY_METRIC_SPECS:
        paths[f"{spec.slug}_spatial"] = spatial_dir / f"{spec.key}.tif"
        paths[f"{spec.slug}_dots"] = point_dir / f"{spec.key}.tif"
        paths[f"{spec.slug}_dots_labeled"] = point_label_dir / f"{spec.key}.tif"
        paths[f"{spec.slug}_threshold"] = threshold_dir / f"{spec.key}.tif"
    return paths


def remove_legacy_quality_render_outputs(extract_dir: Path) -> None:
    for figure_dir in (
        get_figure_spatialmap_dir(extract_dir),
        get_figure_pointmap_dir(extract_dir),
        get_figure_pointmap_label_dir(extract_dir),
        get_figure_thresholdstack_dir(extract_dir),
    ):
        if not figure_dir.is_dir():
            continue
        for path in figure_dir.iterdir():
            if path.is_file():
                path.unlink()


def _component_core_coordinates_and_weights(spatial, patch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_idx = spatial.indices.astype(np.int64, copy=False)
    weights = spatial.data.astype(np.float32, copy=False)
    ys_full = flat_idx % patch.h
    xs_full = flat_idx // patch.h
    core_mask = (
        (ys_full >= patch.core_ly0)
        & (ys_full < patch.core_ly1)
        & (xs_full >= patch.core_lx0)
        & (xs_full < patch.core_lx1)
    )
    ys_core = ys_full[core_mask] - patch.core_ly0
    xs_core = xs_full[core_mask] - patch.core_lx0
    weights_core = weights[core_mask]
    return (
        ys_core.astype(np.int32, copy=False),
        xs_core.astype(np.int32, copy=False),
        weights_core.astype(np.float32, copy=False),
    )


def _component_outline_coordinates(
    spatial,
    patch,
    support_threshold_rel: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ys_core, xs_core, weights_core = _component_core_coordinates_and_weights(spatial, patch)
    if ys_core.size == 0:
        raise ValueError(f"Component has no support inside patch core: {patch.patch_name}")

    peak_index = int(np.argmax(weights_core))
    peak_y_core = int(ys_core[peak_index])
    peak_x_core = int(xs_core[peak_index])

    core_mask = np.zeros((patch.core_h, patch.core_w), dtype=bool)
    threshold = float(np.max(weights_core)) * float(support_threshold_rel)
    core_mask[ys_core, xs_core] = weights_core >= threshold
    if not core_mask[peak_y_core, peak_x_core]:
        core_mask[peak_y_core, peak_x_core] = True

    labeled, _ = ndi.label(core_mask)
    peak_label = int(labeled[peak_y_core, peak_x_core])
    if peak_label > 0:
        core_mask = labeled == peak_label

    eroded = ndi.binary_erosion(core_mask, structure=np.ones((3, 3), dtype=bool))
    outline = core_mask & ~eroded
    if not np.any(outline):
        outline[peak_y_core, peak_x_core] = True

    outline_y_core, outline_x_core = np.nonzero(outline)
    outline_y = outline_y_core.astype(np.int32, copy=False) + int(patch.y0)
    outline_x = outline_x_core.astype(np.int32, copy=False) + int(patch.x0)
    return (
        outline_y.astype(np.int32, copy=False),
        outline_x.astype(np.int32, copy=False),
        int(patch.y0 + peak_y_core),
        int(patch.x0 + peak_x_core),
    )


def _parse_bool_csv(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_profile_metrics(profile_path: Path) -> dict[int, dict[str, float | bool]]:
    rows: dict[int, dict[str, float | bool]] = {}
    with profile_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            component_index = int(row["component_index"])
            metrics: dict[str, float | bool] = {}
            for key in QUALITY_PROFILE_KEYS:
                if key == "r_value_unreliable_joint_only":
                    metrics[key] = _parse_bool_csv(row[key])
                else:
                    metrics[key] = float(row[key])
            rows[component_index] = metrics
    return rows


def _prepare_quality_render_patches(
    y_load_fold_cfg: Any,
    movie_paths: list[Path],
    extract_dir: Path,
):
    from .crop import load_crop_params
    from .save import _collect_patch_meta

    if omegaconf.OmegaConf.is_config(y_load_fold_cfg):
        raw_input = omegaconf.OmegaConf.to_container(y_load_fold_cfg, resolve=True)
    else:
        raw_input = y_load_fold_cfg
    raw_input_path = Path(str(raw_input)).resolve()

    model_dir = get_model_dir(extract_dir)
    if len(movie_paths) == 1 and not raw_input_path.is_dir():
        y_load_fold = movie_paths[0].parent
    else:
        y_load_fold = raw_input_path

    patches = _collect_patch_meta(y_load_fold, model_dir)
    crop_load_fold = extract_dir.parent / "crop"

    if crop_load_fold.is_dir():
        crop_params = load_crop_params(crop_load_fold)
        core_specs = {
            spec.name: (int(spec.y0), int(spec.y1), int(spec.x0), int(spec.x1))
            for spec in crop_params["patch_specs"]
        }
        full_h = max(v[1] for v in core_specs.values())
        full_w = max(v[3] for v in core_specs.values())
        for patch in patches:
            if patch.patch_name not in core_specs:
                raise KeyError(f"Missing crop spec for patch: {patch.patch_name}")
            y0, y1, x0, x1 = core_specs[patch.patch_name]
            core_h = int(y1 - y0)
            core_w = int(x1 - x0)
            pad_h = int(patch.h - core_h)
            pad_w = int(patch.w - core_w)
            if core_h <= 0 or core_w <= 0:
                raise ValueError(f"Invalid core size for {patch.patch_name}: {(core_h, core_w)}")
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"Core larger than patch for {patch.patch_name}: patch={(patch.h, patch.w)}, core={(core_h, core_w)}"
                )
            if (pad_h % 2) != 0 or (pad_w % 2) != 0:
                raise ValueError(
                    f"Expected symmetric pad for {patch.patch_name}: patch={(patch.h, patch.w)}, core={(core_h, core_w)}"
                )
            patch.y0, patch.y1, patch.x0, patch.x1 = y0, y1, x0, x1
            patch.core_h, patch.core_w = core_h, core_w
            patch.core_ly0 = pad_h // 2
            patch.core_ly1 = patch.core_ly0 + core_h
            patch.core_lx0 = pad_w // 2
            patch.core_lx1 = patch.core_lx0 + core_w
        return patches, int(full_h), int(full_w)

    if len(patches) != 1:
        raise FileNotFoundError(
            f"Expected crop para at {crop_para_path} for multi-patch quality rendering."
        )

    patch = patches[0]
    patch.y0, patch.y1, patch.x0, patch.x1 = 0, patch.h, 0, patch.w
    patch.core_h, patch.core_w = patch.h, patch.w
    patch.core_ly0, patch.core_ly1 = 0, patch.h
    patch.core_lx0, patch.core_lx1 = 0, patch.w
    return patches, int(patch.h), int(patch.w)


def _collect_quality_render_components(
    patches,
    extract_dir: Path,
) -> list[QualityRenderComponent]:
    components: list[QualityRenderComponent] = []
    model_dir = get_model_dir(extract_dir)
    profile_dir = get_profile_dir(extract_dir)

    for patch in patches:
        model_path = model_dir / f"{patch.patch_name}.tif.hdf5"
        profile_path = profile_dir / f"{patch.patch_name}.tif.csv"
        if not model_path.is_file() or not profile_path.is_file():
            continue

        metrics_by_component = _read_profile_metrics(profile_path)
        with silence_stdio():
            cnm_model = load_CNMF(str(model_path), n_processes=1, dview=None)
        A = cnm_model.estimates.A.tocsc()
        if A.shape[1] != len(metrics_by_component):
            raise ValueError(
                f"Profile/model component mismatch for {patch.patch_name}: "
                f"A={A.shape[1]}, csv={len(metrics_by_component)}"
            )

        for component_index in range(A.shape[1]):
            spatial = A.getcol(component_index)
            try:
                outline_y, outline_x, peak_y, peak_x = _component_outline_coordinates(
                    spatial,
                    patch,
                )
            except ValueError:
                continue
            metrics = metrics_by_component[component_index]
            components.append(
                QualityRenderComponent(
                    patch_name=patch.patch_name,
                    component_index=int(component_index),
                    x=int(peak_x),
                    y=int(peak_y),
                    outline_y=outline_y,
                    outline_x=outline_x,
                    metrics=metrics,
                )
            )
    return components


def _estimate_spatial_sigma(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    coords = np.column_stack([xs, ys]).astype(np.float64, copy=False)
    if coords.shape[0] < 2:
        return float(QUALITY_SPATIAL_SIGMA_FLOOR_PX), 0.0
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nn = np.asarray(distances[:, 1], dtype=np.float64)
    median_nn = float(np.median(nn))
    sigma_px = float(max(QUALITY_SPATIAL_SIGMA_FLOOR_PX, QUALITY_SPATIAL_SIGMA_FACTOR * median_nn))
    return sigma_px, median_nn


def _choose_adaptive_spatial_sigmas(median_nn_px: float) -> tuple[float, float, float]:
    fine = max(6.0, 1.0 * median_nn_px)
    medium = max(12.0, 2.0 * median_nn_px)
    coarse = max(24.0, 4.0 * median_nn_px)
    if medium <= fine:
        medium = fine * 1.75
    if coarse <= medium:
        coarse = medium * 2.0
    return float(fine), float(medium), float(coarse)


def _build_kernel_regression_field(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    num = np.zeros((h, w), dtype=np.float32)
    den = np.zeros((h, w), dtype=np.float32)
    np.add.at(num, (ys.astype(np.int64), xs.astype(np.int64)), values.astype(np.float32))
    np.add.at(den, (ys.astype(np.int64), xs.astype(np.int64)), 1.0)
    num_blur = ndi.gaussian_filter(num, sigma=float(sigma_px), mode="nearest")
    den_blur = ndi.gaussian_filter(den, sigma=float(sigma_px), mode="nearest")
    field = num_blur / np.maximum(den_blur, 1e-8)
    return field.astype(np.float32, copy=False), den_blur.astype(np.float32, copy=False)


def _build_adaptive_multiscale_field(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    sigmas: tuple[float, float, float],
) -> np.ndarray:
    fine_sigma, medium_sigma, coarse_sigma = sigmas
    field_fine, density_fine = _build_kernel_regression_field(xs, ys, values, shape, fine_sigma)
    field_medium, density_medium = _build_kernel_regression_field(xs, ys, values, shape, medium_sigma)
    field_coarse, density_coarse = _build_kernel_regression_field(xs, ys, values, shape, coarse_sigma)

    sample_mask = np.zeros(shape, dtype=bool)
    sample_mask[ys.astype(np.int64), xs.astype(np.int64)] = True
    dist_px = ndi.distance_transform_edt(~sample_mask).astype(np.float32, copy=False)

    d1 = float(medium_sigma)
    d2 = float(coarse_sigma)
    w_fine = np.clip(1.0 - dist_px / max(d1, 1e-6), 0.0, 1.0).astype(np.float32)
    w_coarse = np.clip((dist_px - d1) / max(d2 - d1, 1e-6), 0.0, 1.0).astype(np.float32)
    w_medium = np.clip(1.0 - w_fine - w_coarse, 0.0, 1.0).astype(np.float32)

    field = (
        w_fine * field_fine
        + w_medium * field_medium
        + w_coarse * field_coarse
    ).astype(np.float32, copy=False)
    return field


def _prepare_metric_space_values(
    spec: QualityMetricSpec,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if finite.size == 0:
        raise RuntimeError(f"No finite values available for metric transform: {spec.key}")

    meta: dict[str, float | int | str] = {"scale": spec.scale}
    if spec.scale == "linear":
        return arr.copy(), finite.astype(np.float64, copy=False), meta

    if spec.scale == "log":
        positive = finite[finite > 0]
        if positive.size == 0:
            raise RuntimeError(
                f"No positive finite values available for log transform: {spec.key}"
            )
        floor = float(np.min(positive))
        transformed = np.empty_like(arr, dtype=np.float64)
        positive_mask = np.isfinite(arr) & (arr > 0)
        transformed[positive_mask] = np.log10(arr[positive_mask])
        transformed[~positive_mask] = np.log10(floor)
        meta["log_floor_value"] = floor
        meta["nonpositive_count"] = int(np.count_nonzero(finite <= 0))
        return transformed, transformed[np.isfinite(transformed)], meta

    raise ValueError(f"Unsupported metric render scale: {spec.scale}")


def _fit_metric_distribution(
    spec: QualityMetricSpec,
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, dict[str, float | int | str]]:
    metric_space_values, fit_values, meta = _prepare_metric_space_values(spec, values)
    fit_mean, fit_std = scipy_norm.fit(fit_values.astype(np.float64, copy=False))
    fit_mean = float(fit_mean)
    fit_std = float(fit_std)
    if not np.isfinite(fit_std) or fit_std <= 0.0:
        fit_std = 1.0
    if np.any(~np.isfinite(metric_space_values)):
        metric_space_values = metric_space_values.copy()
        metric_space_values[~np.isfinite(metric_space_values)] = fit_mean
    return metric_space_values, fit_mean, fit_std, meta


def _build_boolean_metric_render_values(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, mcolors.Normalize]:
    binary = np.asarray(values, dtype=np.float64)
    render_values = np.clip(binary, 0.0, 1.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    return binary, render_values, norm


def _build_metric_render_values(
    spec: QualityMetricSpec,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, mcolors.Normalize, float, float, dict[str, float | int | str]]:
    if spec.scale == "boolean":
        binary, render_values, norm = _build_boolean_metric_render_values(values)
        return binary, render_values, norm, 0.5, 0.5, {"scale": spec.scale}

    metric_space_values, fit_mean, fit_std, meta = _fit_metric_distribution(spec, values)
    half = float(QUALITY_GAUSSIAN_DISPLAY_SIGMA_RANGE * fit_std)
    vmin = float(fit_mean - half)
    vmax = float(fit_mean + half)
    render_values = np.clip(metric_space_values, vmin, vmax)
    if vmin < fit_mean < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=fit_mean, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return metric_space_values, render_values, norm, fit_mean, fit_std, meta


def _metric_bad_side_is_high(spec: QualityMetricSpec) -> bool:
    return spec.key in QUALITY_THRESHOLD_BAD_HIGH_KEYS


def _build_threshold_stack_for_metric(
    spec: QualityMetricSpec,
    components: list[QualityRenderComponent],
    metric_space_values: np.ndarray,
    fit_mean: float,
    fit_std: float,
    full_h: int,
    full_w: int,
    z_values: np.ndarray,
) -> np.ndarray:
    if spec.scale == "boolean":
        frame = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        for component, raw_value in zip(components, metric_space_values, strict=False):
            color = (
                QUALITY_THRESHOLD_REJECT_RGB
                if bool(raw_value)
                else QUALITY_THRESHOLD_JOIN_ACCEPT_RGB
            )
            frame[component.outline_y, component.outline_x] = color
        return frame[None, ...]

    bad_is_high = _metric_bad_side_is_high(spec)
    if bad_is_high:
        oriented_z = -((metric_space_values - fit_mean) / fit_std)
    else:
        oriented_z = (metric_space_values - fit_mean) / fit_std

    frames: list[np.ndarray] = []
    for z_thr in z_values:
        frame = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        reject_mask = oriented_z < float(z_thr)
        for component, is_reject in zip(components, reject_mask, strict=False):
            color = QUALITY_THRESHOLD_REJECT_RGB if bool(is_reject) else QUALITY_THRESHOLD_ACCEPT_RGB
            frame[component.outline_y, component.outline_x] = color
        frames.append(frame)
    return np.stack(frames, axis=0)


def _format_threshold_value(value: float) -> str:
    abs_v = abs(float(value))
    if abs_v >= 100.0 or (0.0 < abs_v < 0.01):
        return f"{value:.6e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _build_threshold_stack_labels(
    spec: QualityMetricSpec,
    fit_mean: float,
    fit_std: float,
    z_values: np.ndarray,
) -> list[str] | None:
    if spec.scale == "boolean":
        return None

    bad_is_high = _metric_bad_side_is_high(spec)
    labels: list[str] = []
    for z_thr in z_values:
        metric_thr = (
            fit_mean - float(z_thr) * fit_std
            if bad_is_high
            else fit_mean + float(z_thr) * fit_std
        )
        op = ">" if bad_is_high else "<"
        labels.append(f"reject: value {op} {_format_threshold_value(metric_thr)}")
    return labels


def _write_rgb_threshold_stack(
    stack: np.ndarray,
    out_path: Path,
    labels: list[str] | None = None,
) -> None:
    metadata: dict[str, object] = {"axes": "ZYXS"}
    if labels is not None:
        metadata["Labels"] = labels
    tifffile.imwrite(
        out_path,
        stack,
        photometric="rgb",
        compression="zlib",
        imagej=True,
        metadata=metadata,
    )


def _format_metric_label_texts(
    spec: QualityMetricSpec,
    values: np.ndarray,
) -> list[str]:
    if spec.scale == "boolean":
        return ["1" if bool(v) else "0" for v in values]

    labels: list[str] = []
    for value in values:
        v = float(value)
        abs_v = abs(v)
        if abs_v >= 100.0 or (0.0 < abs_v < 0.01):
            labels.append(f"{v:.2e}")
        else:
            labels.append(f"{v:.2f}")
    return labels


def _render_canvas_rgba(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return rgba


def _resolve_metric_cmap(cmap: Any):
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def _render_metric_dots_tif(
    xs: np.ndarray,
    ys: np.ndarray,
    display_values: np.ndarray,
    label_texts: list[str] | None,
    out_path: Path,
    full_h: int,
    full_w: int,
    cmap_name: str,
    norm: mcolors.Normalize,
) -> None:
    fig = plt.figure(
        figsize=(
            full_w * QUALITY_RENDER_SCALE_FACTOR / QUALITY_RENDER_DPI,
            full_h * QUALITY_RENDER_SCALE_FACTOR / QUALITY_RENDER_DPI,
        ),
        dpi=QUALITY_RENDER_DPI,
        frameon=False,
    )
    fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    cmap = _resolve_metric_cmap(cmap_name)
    ax.scatter(
        xs,
        ys,
        c=display_values,
        cmap=cmap,
        norm=norm,
        s=QUALITY_DOT_SIZE,
        alpha=QUALITY_DOT_ALPHA,
        linewidths=0.0,
        marker="o",
    )
    if label_texts is not None:
        rgba = cmap(norm(display_values))
        colors = rgba[:, :3]
        stroke = [mpe.withStroke(linewidth=0.6, foreground="black")]
        for x, y, label_text, color in zip(xs, ys, label_texts, colors, strict=False):
            ax.text(
                float(x) + float(QUALITY_LABEL_OFFSET_X),
                float(y) + float(QUALITY_LABEL_OFFSET_Y),
                str(label_text),
                color=color,
                fontsize=QUALITY_LABEL_FONT_SIZE,
                ha="left",
                va="center",
                clip_on=True,
                path_effects=stroke,
            )
    ax.set_xlim(-0.5, full_w - 0.5)
    ax.set_ylim(full_h - 0.5, -0.5)
    ax.set_axis_off()
    rgba = _render_canvas_rgba(fig)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = np.round(rgba[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    rgb[rgba[..., 3] == 0] = 0
    tifffile.imwrite(out_path, rgb, photometric="rgb")


def _render_metric_spatial_map_tif(
    rgb: np.ndarray,
    out_path: Path,
) -> None:
    tifffile.imwrite(out_path, rgb, photometric="rgb")


def render_extract_quality_outputs(
    y_load_fold_cfg: Any,
    movie_paths: list[Path],
    extract_dir: Path,
    logger: logging.Logger,
    progress_bar: tqdm.tqdm | None = None,
) -> None:
    if not movie_paths:
        return

    get_figure_spatialmap_dir(extract_dir).mkdir(parents=True, exist_ok=True)
    get_figure_pointmap_dir(extract_dir).mkdir(parents=True, exist_ok=True)
    get_figure_pointmap_label_dir(extract_dir).mkdir(parents=True, exist_ok=True)
    get_figure_thresholdstack_dir(extract_dir).mkdir(parents=True, exist_ok=True)
    remove_legacy_quality_render_outputs(extract_dir)
    output_paths = get_quality_render_output_paths(extract_dir)
    patches, full_h, full_w = _prepare_quality_render_patches(
        y_load_fold_cfg=y_load_fold_cfg,
        movie_paths=movie_paths,
        extract_dir=extract_dir,
    )
    components = _collect_quality_render_components(patches, extract_dir)
    if not components:
        raise RuntimeError("No component points were available for extract quality rendering.")

    xs = np.asarray([component.x for component in components], dtype=np.float64)
    ys = np.asarray([component.y for component in components], dtype=np.float64)
    sigma_px, median_nn_px = _estimate_spatial_sigma(xs, ys)
    fine_sigma_px, medium_sigma_px, coarse_sigma_px = _choose_adaptive_spatial_sigmas(median_nn_px)
    for spec in QUALITY_METRIC_SPECS:
        metric_values = np.asarray(
            [component.metrics[spec.key] for component in components],
            dtype=np.float64,
        )
        metric_space_values, display_values, norm, fit_mean, fit_std, meta = _build_metric_render_values(
            spec,
            metric_values,
        )
        field = _build_adaptive_multiscale_field(
            xs=xs,
            ys=ys,
            values=metric_space_values,
            shape=(full_h, full_w),
            sigmas=(fine_sigma_px, medium_sigma_px, coarse_sigma_px),
        )
        field_for_render = np.clip(field, float(norm.vmin), float(norm.vmax))
        rgb = np.asarray(
            _resolve_metric_cmap(spec.cmap)(
                norm(field_for_render),
                bytes=True,
            )[..., :3],
            dtype=np.uint8,
        )
        _render_metric_spatial_map_tif(rgb, output_paths[f"{spec.slug}_spatial"])
        if progress_bar is not None:
            progress_bar.update(1)
        _render_metric_dots_tif(
            xs=xs,
            ys=ys,
            display_values=display_values,
            label_texts=None,
            out_path=output_paths[f"{spec.slug}_dots"],
            full_h=full_h,
            full_w=full_w,
            cmap_name=spec.cmap,
            norm=norm,
        )
        if progress_bar is not None:
            progress_bar.update(1)
        _render_metric_dots_tif(
            xs=xs,
            ys=ys,
            display_values=display_values,
            label_texts=_format_metric_label_texts(spec, metric_space_values),
            out_path=output_paths[f"{spec.slug}_dots_labeled"],
            full_h=full_h,
            full_w=full_w,
            cmap_name=spec.cmap,
            norm=norm,
        )
        if progress_bar is not None:
            progress_bar.update(1)
        threshold_stack = _build_threshold_stack_for_metric(
            spec=spec,
            components=components,
            metric_space_values=metric_space_values,
            fit_mean=fit_mean,
            fit_std=fit_std,
            full_h=full_h,
            full_w=full_w,
            z_values=QUALITY_THRESHOLD_Z_VALUES,
        )
        threshold_labels = _build_threshold_stack_labels(
            spec=spec,
            fit_mean=fit_mean,
            fit_std=fit_std,
            z_values=QUALITY_THRESHOLD_Z_VALUES,
        )
        _write_rgb_threshold_stack(
            threshold_stack,
            output_paths[f"{spec.slug}_threshold"],
            labels=threshold_labels,
        )
        if progress_bar is not None:
            progress_bar.update(1)
    logger.info(
        "Saved extract quality renders to %s: %s",
        extract_dir,
        ", ".join(path.name for path in output_paths.values()),
    )


class Extract:
    def __init__(
        self,
        y_load_fold: str,
        extract_save_fold: str | None,
        data: Any,
        init: Any,
        motion: Any,
        preprocess: Any,
        temporal: Any,
        spatial: Any,
        patch: Any,
        merging: Any,
        enable: bool = False,
        **kwargs,
    ) -> None:
        self.enable = bool(enable)
        cfg_dict: dict[str, Any] = {
            "extract": {
                "y_load_fold": _to_container(y_load_fold),
                "extract_save_fold": _to_container(extract_save_fold),
            },
            "data": _to_container(data),
            "init": _to_container(init),
            "motion": _to_container(motion),
            "preprocess": _to_container(preprocess),
            "temporal": _to_container(temporal),
            "spatial": _to_container(spatial),
            "patch": _to_container(patch),
            "merging": _to_container(merging),
        }
        self.cfg = omegaconf.OmegaConf.create(cfg_dict)

    def forward(self) -> None:
        configure_warning_filters()
        logger = setup_logger()
        movie_paths, auto_extract_dir = resolve_input_movies(self.cfg.extract.y_load_fold)
        extract_dir = resolve_extract_dir(self.cfg.extract.extract_save_fold, auto_extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        get_model_dir(extract_dir).mkdir(parents=True, exist_ok=True)
        get_profile_dir(extract_dir).mkdir(parents=True, exist_ok=True)
        get_figure_spatialmap_dir(extract_dir).mkdir(parents=True, exist_ok=True)
        get_figure_pointmap_dir(extract_dir).mkdir(parents=True, exist_ok=True)
        get_figure_pointmap_label_dir(extract_dir).mkdir(parents=True, exist_ok=True)
        get_figure_thresholdstack_dir(extract_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Resolved %d .tif files. extract dir: %s, model dir: %s, profile dir: %s",
            len(movie_paths),
            extract_dir,
            get_model_dir(extract_dir),
            get_profile_dir(extract_dir),
        )
        pending_items: list[dict[str, Any]] = []
        needs_cluster = False
        model_done_count = 0
        profile_done_count = 0
        for movie_path in movie_paths:
            model_path = get_model_path(movie_path, extract_dir)
            profile_path = get_profile_csv_path(movie_path, extract_dir)
            need_model = not model_path.is_file()
            need_profile = need_model or not profile_path.is_file()
            if model_path.is_file():
                model_done_count += 1
            if profile_path.is_file():
                profile_done_count += 1
            if not need_model and not need_profile:
                continue
            pending_items.append(
                {
                    "movie_path": movie_path,
                    "model_path": model_path,
                    "profile_path": profile_path,
                    "need_model": need_model,
                    "need_profile": need_profile,
                }
            )
            needs_cluster = needs_cluster or need_model

        dview = None
        n_processes = 1
        if needs_cluster:
            with silence_stdio():
                _, dview, n_processes = caiman.cluster.setup_cluster()
        model_bar = tqdm.tqdm(
            total=len(movie_paths),
            initial=model_done_count,
            desc="extract(model)",
            unit="patch",
            dynamic_ncols=True,
            position=0,
        )
        profile_bar = tqdm.tqdm(
            total=len(movie_paths),
            initial=profile_done_count,
            desc="extract(profile)",
            unit="patch",
            dynamic_ncols=True,
            position=1,
        )
        try:
            for item in pending_items:
                movie_path = item["movie_path"]
                if item["need_model"]:
                    run_single_patch(
                        movie_path,
                        self.cfg,
                        dview,
                        n_processes,
                        extract_dir,
                        logger,
                        profile_path=item["profile_path"] if item["need_profile"] else None,
                    )
                    model_bar.update(1)
                    if item["need_profile"]:
                        profile_bar.update(1)
                    continue

                with silence_stdio():
                    cnm_model = load_CNMF(
                        str(item["model_path"]),
                        n_processes=1,
                        dview=None,
                    )
                movie_tyx = load_movie_tyx(movie_path)
                with silence_stdio():
                    save_neuron_profile_csv(
                        cnm=cnm_model,
                        movie_path=movie_path,
                        model_path=item["model_path"],
                        profile_path=item["profile_path"],
                        movie_tyx=movie_tyx,
                        logger=logger,
                    )
                profile_bar.update(1)
        finally:
            if dview is not None:
                with silence_stdio():
                    caiman.stop_server(dview=dview)
            model_bar.close()
            profile_bar.close()
        figure_total = len(get_quality_render_output_paths(extract_dir)) + 1
        figure_bar = tqdm.tqdm(
            total=figure_total,
            desc="extract(figure)",
            dynamic_ncols=True,
            position=0,
        )
        try:
            render_extract_quality_outputs(
                y_load_fold_cfg=self.cfg.extract.y_load_fold,
                movie_paths=movie_paths,
                extract_dir=extract_dir,
                logger=logger,
                progress_bar=figure_bar,
            )
            write_extract_stats_json(extract_dir, logger)
            render_extract_stats_png(extract_dir, logger)
            figure_bar.update(1)
        finally:
            figure_bar.close()
