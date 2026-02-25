import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import omegaconf
import tqdm

import caiman
import caiman.cluster
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params


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
        return [movie_path], movie_path.parent / "model"

    if input_path.is_dir():
        movie_paths = sorted(
            [
                p.resolve()
                for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() == ".tif"
            ],
            key=lambda p: p.name.lower(),
        )
        if not movie_paths:
            raise ValueError(f"No .tif files found in input directory: {input_path}")
        return movie_paths, input_path.resolve().parent / "model"

    raise FileNotFoundError(f"Input path does not exist: {raw}")


def resolve_model_dir(model_save_fold_cfg: Any, auto_model_dir: Path) -> Path:
    if model_save_fold_cfg is None:
        return auto_model_dir

    if omegaconf.OmegaConf.is_config(model_save_fold_cfg):
        raw = omegaconf.OmegaConf.to_container(model_save_fold_cfg, resolve=True)
    else:
        raw = model_save_fold_cfg
    if not isinstance(raw, str):
        raise TypeError(
            f"extract.model_save_fold must be a path string or null, got {type(raw).__name__}"
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
    model_dir: Path,
    logger: logging.Logger,
) -> None:
    opts = params.CNMFParams(params_dict=build_params_dict(cfg, movie_path))
    bord_px = 0
    memmap_base_name = f"memmap_{movie_path.stem}_{os.getpid()}_{uuid4().hex[:8]}_"

    if bool(cfg.motion.enable):
        mc = MotionCorrect(opts.data["fnames"], dview=dview, **opts.get_group("motion"))
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

    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
    cnm.fit(images)
    save_model(cnm, movie_path, model_dir, logger)


def save_model(
    cnm: cnmf.CNMF,
    fname: Path,
    model_dir: Path,
    logger: logging.Logger,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / f"{fname.name}.hdf5"
    cnm.save(str(model_path))
    logger.info("[%s] model saved: %s", fname.name, model_path)


class Extract:
    def __init__(
        self,
        y_load_fold: str,
        model_save_fold: str | None,
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
                "model_save_fold": _to_container(model_save_fold),
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
        logger = setup_logger()
        movie_paths, auto_model_dir = resolve_input_movies(self.cfg.extract.y_load_fold)
        model_dir = resolve_model_dir(self.cfg.extract.model_save_fold, auto_model_dir)
        logger.info("Resolved %d .tif files. model dir: %s", len(movie_paths), model_dir)

        with silence_stdio():
            _, dview, n_processes = caiman.cluster.setup_cluster()
        try:
            for movie_path in tqdm.tqdm(
                movie_paths,
                desc="extract",
                unit="patch",
                dynamic_ncols=True,
            ):
                with silence_stdio():
                    run_single_patch(
                        movie_path,
                        self.cfg,
                        dview,
                        n_processes,
                        model_dir,
                        logger,
                    )
        finally:
            with silence_stdio():
                caiman.stop_server(dview=dview)
