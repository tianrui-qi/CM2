import sys
import hydra
import omegaconf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web import run_app


def _resolve_optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def _resolve_app_paths(cfg: omegaconf.DictConfig) -> tuple[Path, Path, Path]:
    data_fold = _resolve_optional_path(cfg.data_fold)

    bg_load_path = _resolve_optional_path(cfg.bg_load_path)
    extract_load_fold = _resolve_optional_path(cfg.extract_load_fold)
    app_fold = _resolve_optional_path(cfg.app_fold)

    if data_fold is not None:
        bg_load_path = bg_load_path or (data_fold / "save" / "Ybandpass.tif")
        extract_load_fold = extract_load_fold or (data_fold / "extract")
        app_fold = app_fold or (data_fold / "app")

    missing = [
        name
        for name, value in (
            ("bg_load_path", bg_load_path),
            ("extract_load_fold", extract_load_fold),
            ("app_fold", app_fold),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing required app path config. "
            "Provide data_fold, or set all of bg_load_path, extract_load_fold, and app_fold. "
            f"Missing: {', '.join(missing)}"
        )

    return bg_load_path, extract_load_fold, app_fold


@hydra.main(
    version_base=None,
    config_path="../config", config_name="pipeline/app",
)
def main(cfg: omegaconf.DictConfig) -> None:
    bg_load_path, extract_load_fold, app_fold = _resolve_app_paths(cfg)
    run_app(
        bg_load_path=bg_load_path,
        extract_load_fold=extract_load_fold,
        app_fold=app_fold,
        host=str(cfg.host),
        port=int(cfg.port),
        debug=bool(cfg.debug),
        force_rebuild=bool(cfg.force_rebuild),
    )


if __name__ == "__main__": main()
