import hydra
import omegaconf

import src


@hydra.main(
    version_base=None, 
    config_path="../config", config_name="pipeline/process"
)
def main(cfg: omegaconf.DictConfig) -> None:
    if cfg.recon.enable: src.Recon(**cfg.recon).forward()
    if cfg.normalize.enable: src.Normalize(**cfg.normalize).forward()
    if cfg.crop.enable: src.Crop(**cfg.crop).forward()


if __name__ == "__main__": main()
