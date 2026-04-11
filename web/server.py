from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

from flask import Flask, abort, send_file, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def create_app(app_fold: Path) -> Flask:
    cache_root = app_fold.resolve()
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["CACHE_ROOT"] = str(cache_root)

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/cache/<path:filename>")
    def cache_file(filename: str):
        return send_from_directory(cache_root, filename)

    @app.route("/vendor/plotly.min.js")
    def plotly_vendor():
        path = files("plotly.package_data").joinpath("plotly.min.js")
        return send_file(path, mimetype="application/javascript")

    @app.route("/health")
    def health():
        metadata_path = cache_root / "metadata.json"
        if not metadata_path.is_file():
            abort(503, description="Web cache missing. Run script/app.py first.")
        return {"ok": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the standalone neuron ROI web app.")
    parser.add_argument("--app_fold", type=Path, required=True, help="Path to the app cache folder.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.app_fold)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
