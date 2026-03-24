import hydra
import omegaconf
from pathlib import Path

import src


@hydra.main(
    version_base=None, 
    config_path="../config", config_name="pipeline/process"
)
def main(cfg: omegaconf.DictConfig) -> None:
    # set default values
    if cfg.y_path is None:
        cfg.y_path = str(Path(cfg.x_path).parent / "Y.tif")
    if cfg.stitch.result_save_fold is None:
        cfg.stitch.result_save_fold = str(
            Path(cfg.stitch.y_save_path).parent / "stitch"
        )
    if cfg.vessel.result_save_fold is None:
        cfg.vessel.result_save_fold = str(
            Path(cfg.vessel.y_load_path).parent / "vessel"
        )
    if cfg.crop.y_save_fold is None:
        cfg.crop.y_save_fold = str(
            Path(cfg.crop.y_load_path).with_suffix("")
        )
    if cfg.crop.result_save_fold is None:
        cfg.crop.result_save_fold = str(
            Path(cfg.crop.y_load_path).parent / "crop"
        )
    if cfg.crop.para_load_path is None:
        cfg.crop.para_load_path = str(
            Path(cfg.crop.y_save_fold).parent / "stitch" / "para.json"
        )
    if cfg.extract.model_save_fold is None:
        cfg.extract.model_save_fold = str(
            Path(cfg.extract.y_load_fold).parent / "extract"
        )
    # forward
    if cfg.stitch.enable:
        src.Stitch(**cfg.stitch).forward()
    if cfg.normalize.enable:
        src.Normalize(**cfg.normalize).forward()
    if cfg.crop.enable:
        src.Crop(**cfg.crop).forward()
    if cfg.extract.enable:
        src.Extract(**cfg.extract).forward()
    if cfg.vessel.enable:
        src.Vessel(**cfg.vessel).forward()


if __name__ == "__main__": main()
