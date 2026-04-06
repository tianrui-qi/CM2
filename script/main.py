import hydra
import omegaconf
from pathlib import Path

import src


@hydra.main(
    version_base=None, 
    config_path="../config", config_name="pipeline/main"
)
def main(cfg: omegaconf.DictConfig) -> None:
    # set default values
    if cfg.y_path is None:
        cfg.y_path = str(Path(cfg.x_path).parent / "Y.tif")
    if cfg.stitch.stitch_save_fold is None:
        cfg.stitch.stitch_save_fold = str(
            Path(cfg.stitch.y_save_path).parent / "stitch"
        )
    if cfg.crop.y_save_fold is None:
        cfg.crop.y_save_fold = str(
            Path(cfg.crop.y_load_path).with_suffix("")
        )
    if cfg.crop.crop_save_fold is None:
        cfg.crop.crop_save_fold = str(
            Path(cfg.crop.y_load_path).parent / "crop"
        )
    if cfg.extract.extract_save_fold is None:
        cfg.extract.extract_save_fold = str(
            Path(cfg.extract.y_load_fold).parent / "extract"
        )
    if cfg.save.save_fold is None:
        cfg.save.save_fold = str(
            Path(cfg.save.y_load_path).parent / "save"
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
    if cfg.save.enable:
        src.Save(**cfg.save).forward()


if __name__ == "__main__": main()
