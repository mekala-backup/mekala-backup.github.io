# Less Real is More

This repository contains two parts:

1. A static paper website in `index.html`, `styles.css`, and `site.js`
2. YOLO evaluation and comparison utilities in `app.py` and `yolo_compare.py`

## Repository Layout

- `index.html` - the static paper website
- `styles.css` - visual styling for the site
- `site.js` - chart, gallery, and table interaction logic
- `ratio_vs_map50.png` - hero chart used by the site
- `images in research paper/` - qualitative figures used by the gallery
- `app.py` - main CLI for evaluation, dataset mixing, and model comparison
- `yolo_compare.py` - standalone multi-model comparison script with a config block
- `app_dep.py` - deprecated legacy version of the CLI
- `example_paper_outputs.py` - deprecated example script that imports `app_dep.py`

## Run the static website

The site is plain HTML/CSS/JS. No build step is required.

From the repository root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Run the main Python CLI

`app.py` is the primary script. It supports four commands:

- `evaluate` - evaluate one YOLO segmentation model
- `mix` - create a mixed dataset from two YOLO datasets
- `paper` - generate paper-ready outputs for two models
- `compare` - compare multiple models and write per-model outputs

Example commands:

```bash
python3 app.py evaluate --model /path/to/model.pt --test-data /path/to/test_dataset
python3 app.py mix --dataset-a /path/to/dataset_a --dataset-b /path/to/dataset_b
python3 app.py paper --synth-model /path/to/synth.pt --real-model /path/to/real.pt --test-data /path/to/test_dataset
python3 app.py compare --model /path/to/a.pt --model /path/to/b.pt --test-data /path/to/test_dataset
```

If you run `python3 app.py` with no subcommand, it starts the interactive prompt.

### Expected dataset layout

The YOLO helpers expect a dataset root that contains:

- `images/`
- `labels/`

For `app.py compare`, `paper`, and `evaluate`, the test dataset should point to a YOLO-style dataset root.

## Run the standalone comparison script

`yolo_compare.py` is a self-contained research script. It does not use CLI flags.

To use it:

1. Open the file and edit the `CONFIG` block near the top.
2. Set `DATA_YAML`, `SPLIT`, `MODELS`, and output options.
3. Run:

```bash
python3 yolo_compare.py
```

This script reads the dataset from the YAML file, runs all configured models, and writes comparison outputs under `OUT_ROOT`.

## Deprecated scripts

- `app_dep.py` is deprecated. Use `app.py` instead.
- `example_paper_outputs.py` is deprecated. It only demonstrates the older `app_dep.paper_outputs(...)` workflow.

Do not build new workflows on either file.

## Dependencies

The Python scripts use packages including:

- `ultralytics`
- `torch`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `rich`
- `pyyaml`

Install them in your environment before running the CLI tools.

## Generated files

The repository ignores common generated artifacts such as:

- Python caches and virtual environments
- local logs
- evaluation outputs
- comparison output folders

If you add a new output directory, update `.gitignore` to match it.
