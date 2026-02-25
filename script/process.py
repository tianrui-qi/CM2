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
    if cfg.y_path is None: cfg.y_path = str(
        Path(cfg.x_path).parent / "Y.tif"
    )
    if cfg.crop.y_save_fold is None: cfg.crop.y_save_fold = str(
        Path(cfg.crop.y_load_path).with_suffix("")
    )
    # forward
    if cfg.recon.enable: src.Recon(**cfg.recon).forward()
    if cfg.normalize.enable: src.Normalize(**cfg.normalize).forward()
    if cfg.crop.enable: src.Crop(**cfg.crop).forward()
    if cfg.extract.enable: src.Extract(**cfg.extract).forward()


if __name__ == "__main__": main()