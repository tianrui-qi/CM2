# Web

Standalone neuron ROI web app code.

This folder contains the web modules and static assets. The runtime entrypoint is:

- [script/app.py](/Users/tianrui.qi/Documents/GitHub/CM2/script/app.py)

## Run

```bash
python -m script.app 'data_fold=/path/to/dataset'
```

Then open:

```text
http://127.0.0.1:8765
```

## Notes

- If `data_fold` is set, defaults are inferred as:
  - `bg_load_path = data_fold/save/Ybandpass.tif`
  - `extract_load_fold = data_fold/extract`
  - `app_fold = data_fold/app`
- Explicit `bg_load_path`, `extract_load_fold`, or `app_fold` overrides win over inferred defaults
- If `data_fold` is not set, all of `bg_load_path`, `extract_load_fold`, and `app_fold` are required
- app cache is reused automatically when it matches the current inputs
- otherwise the cache is rebuilt and the server starts
