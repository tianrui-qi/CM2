import hydra
import omegaconf
from pathlib import Path

import src


@hydra.main(
    version_base=None, 
    config_path="../config", config_name="pipeline/save"
)
def main(cfg: omegaconf.DictConfig) -> None:
    # set default values
    if cfg.y_load_path is None:
        cfg.y_load_path = str(Path(cfg.x_load_path).parent / "Y.tif")
    if cfg.model_load_fold is None:
        cfg.model_load_fold = str(Path(cfg.y_load_path).parent / "model")
    if cfg.save_fold is None:
        cfg.save_fold = str(Path(cfg.y_load_path).parent / "save")
    if cfg.para_load_path is None:
        cfg.para_load_path = str(Path(cfg.y_load_path).parent / "crop" / "para.json")
    # forward
    src.Save(**cfg).forward()   # type: ignore


if __name__ == "__main__": main()
