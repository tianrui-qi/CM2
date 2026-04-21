from __future__ import annotations

import json
import logging
from pathlib import Path

from .build_cache import (
    BACKGROUND_DIRNAME,
    METADATA_FILE_NAME,
    POINTS_FILE_NAME,
    TRACE_SOURCE_FILES,
    build_cache,
)
from .server import create_app


def _cache_metadata_path(app_fold: Path) -> Path:
    return app_fold / METADATA_FILE_NAME


def _cache_matches_request(
    *,
    app_fold: Path,
    bg_load_path: Path,
    extract_load_fold: Path,
    crop_load_fold: Path,
) -> bool:
    metadata_path = _cache_metadata_path(app_fold)
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    trace_sources = metadata.get("trace_sources")
    if not isinstance(trace_sources, dict):
        return False

    required_paths = [
        app_fold / POINTS_FILE_NAME,
        app_fold / BACKGROUND_DIRNAME / "background.png",
    ]
    for source_key in TRACE_SOURCE_FILES:
        trace_spec = trace_sources.get(source_key)
        if not isinstance(trace_spec, dict):
            return False
        trace_file = trace_spec.get("file")
        if not isinstance(trace_file, str) or not trace_file:
            return False
        required_paths.append(app_fold / trace_file)
    if not all(path.is_file() for path in required_paths):
        return False

    return (
        Path(metadata.get("extract_load_fold", "")).resolve() == extract_load_fold.resolve()
        and Path(metadata.get("bg_load_path", "")).resolve() == bg_load_path.resolve()
        and Path(metadata.get("crop_load_fold", "")).resolve() == crop_load_fold.resolve()
    )


def run_app(
    *,
    bg_load_path: Path,
    extract_load_fold: Path,
    crop_load_fold: Path,
    app_fold: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    debug: bool = False,
    force_rebuild: bool = False,
) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    bg_load_path = bg_load_path.resolve()
    extract_load_fold = extract_load_fold.resolve()
    crop_load_fold = crop_load_fold.resolve()
    app_fold = app_fold.resolve()

    if force_rebuild or not _cache_matches_request(
        app_fold=app_fold,
        bg_load_path=bg_load_path,
        extract_load_fold=extract_load_fold,
        crop_load_fold=crop_load_fold,
    ):
        build_cache(
            bg_load_path=bg_load_path,
            extract_load_fold=extract_load_fold,
            crop_load_fold=crop_load_fold,
            app_fold=app_fold,
        )

    app = create_app(app_fold)
    app.run(host=host, port=port, debug=debug)
