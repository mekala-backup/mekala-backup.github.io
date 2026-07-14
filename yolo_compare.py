"""
YOLO Multi-Model Comparison Tool for Research Paper
=====================================================
Edit the CONFIG block below, then run:  python yolo_compare.py

No CLI flags needed — everything is set in code.
"""

from __future__ import annotations

import gc
import random
import shutil
import tempfile
import time
import yaml as yaml_lib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
# Cap PyTorch's CPU thread usage. Without this, PyTorch aggressively uses all
# available cores, which on thin/fanless laptops can cause thermal throttling
# or a power-related hard shutdown rather than a clean OOM crash. Lower this
# further (e.g. 2) if the system is still unstable.
torch.set_num_threads(4)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

console = Console()

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — edit this block
# ══════════════════════════════════════════════════════════════════════════

# Path to your dataset's data.yaml (the one with path:/train:/val:/test:/nc:/names:)
DATA_YAML = "/home/mekala/Desktop/evals/real_test/transparent object with and without cap without clutter.v5i.yolo26/data.yaml"

# Which split inside that yaml to evaluate on: "train", "val", or "test"
SPLIT = "test"

# Models to compare — (path_to_pt, display_name)
MODELS: list[tuple[str, str]] = [
    ("/home/mekala/Desktop/segment/0_100_synth_real/weights/best.pt",  "real"),
    ("/home/mekala/Desktop/segment/5_95_synth_real/weights/best.pt",  "95%"),
    ("/home/mekala/Desktop/segment/10_90_synth_real/weights/best.pt",  "90%"),
    ("/home/mekala/Desktop/segment/25_75_synth_real/weights/best.pt", "75%"),
    ("/home/mekala/Desktop/segment/50_50_synth_real/weights/best.pt", "50%"),
    ("/home/mekala/Desktop/segment/75_25_synth_real/weights/best.pt",  "25%"),
    ("/home/mekala/Desktop/segment/90_10_synth_real/weights/best.pt",  "10%"),
    ("/home/mekala/Desktop/segment/95_5_synth_real/weights/best.pt",  "5%"),
    ("/home/mekala/Desktop/segment/100_0_synth_real/weights/best.pt",  "Synthetic"),
]

IMGSZ = 640
CONF = 0.25
IOU_THRESHOLD = 0.5
OUT_ROOT = "paper_results_all"

# Lower this if you hit memory issues / crashes. 4-8 is safe for 16GB RAM, CPU-only laptops.
BATCH_SIZE = 6

# How many annotated prediction images to write to disk per model.
# Set to None to save all of them; lower this if disk/RAM is tight.
MAX_SAVED_PREDICTIONS = None

# Number of sample test images to use for the side-by-side "original vs every
# model" grid comparison. Each one becomes its own grid image. Set to None to
# use ALL test images (slower/heavier — runs one extra full-dataset inference
# pass per model). If your system is crashing, keep this small (e.g. 30-50)
# until you confirm the main run is stable, then increase.
GRID_COMPARISON_SAMPLES = None

# ══════════════════════════════════════════════════════════════════════════
# End of config — implementation below
# ══════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


@dataclass
class DatasetInfo:
    yaml_path: Path
    root: Path
    images_dir: Path
    labels_dir: Path
    nc: int
    names: list[str]


def load_dataset_info(yaml_path: str | Path, split: str) -> DatasetInfo:
    """Read the user's own data.yaml directly — no regeneration."""
    yaml_path = _path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml_lib.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Could not parse {yaml_path} as a YOLO data.yaml (got: {type(data)})")

    base = Path(data["path"]).expanduser() if "path" in data and data["path"] else yaml_path.parent
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()

    split_rel = data.get(split)
    if split_rel is None:
        raise KeyError(f"Split '{split}' not found in {yaml_path}. Available: train/val/test")

    images_dir = (base / split_rel).resolve()
    # labels dir mirrors images dir but with "labels" instead of "images" in the path
    labels_dir = Path(str(images_dir).replace("/images", "/labels"))

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    nc = int(data.get("nc", 1))
    names = data.get("names", [f"class_{i}" for i in range(nc)])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    return DatasetInfo(
        yaml_path=yaml_path, root=base, images_dir=images_dir,
        labels_dir=labels_dir, nc=nc, names=names,
    )


def _label_path_for(images_dir: Path, labels_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return labels_dir / rel.with_suffix(".txt")


def _yolo_label_boxes(label_path: Path, image_shape: tuple[int, int]) -> list[list[float]]:
    if not label_path.is_file():
        return []
    height, width = image_shape
    boxes: list[list[float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = parts[:5]
        xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
        boxes.append([
            max(0.0, xc - w / 2), max(0.0, yc - h / 2),
            min(float(width), xc + w / 2), min(float(height), yc + h / 2),
        ])
    return boxes


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_pil(image_array: Any) -> Image.Image:
    if isinstance(image_array, Image.Image):
        return image_array.convert("RGB")
    return Image.fromarray(image_array[:, :, ::-1]).convert("RGB")


def _make_grid(images: list[Image.Image], path: Path, cols: int = 3) -> Path:
    if not images:
        raise ValueError("No images for grid.")
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    cell_w = min(max(img.width for img in images), 480)
    cell_h = min(max(img.height for img in images), 480)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        fitted = ImageOps.contain(img, (cell_w, cell_h))
        cell = Image.new("RGB", (cell_w, cell_h), "white")
        cell.paste(fitted, ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2))
        canvas.paste(cell, (col * cell_w, row * cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _make_comparison_grid(
    original_bgr: np.ndarray,
    model_frames: list[tuple[str, np.ndarray]],
    cell_size: int = 320,
) -> np.ndarray:
    """Builds one row-major grid: first cell is the original image labeled
    'Original', followed by one cell per model labeled with its name.
    Returns a BGR image ready for cv2.imwrite."""
    cells: list[np.ndarray] = []
    for label, frame in [("Original", original_bgr)] + model_frames:
        resized = cv2.resize(frame, (cell_size, cell_size))
        cv2.putText(resized, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cells.append(resized)

    cols = min(4, len(cells))
    rows = (len(cells) + cols - 1) // cols
    grid = np.full((rows * cell_size, cols * cell_size, 3), 255, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        grid[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = cell
    return grid


def _batch_predict_stream(model: YOLO, image_paths: list[Path], imgsz: int, conf: float,
                           batch_size: int = 4):
    """Yields (image_path, result) pairs one batch at a time.
    Does NOT hold all results in memory at once — critical for low-RAM machines."""
    paths_str = [str(p) for p in image_paths]
    for i in range(0, len(paths_str), batch_size):
        chunk_paths = image_paths[i:i + batch_size]
        chunk_str = paths_str[i:i + batch_size]
        chunk_results = model.predict(source=chunk_str, imgsz=imgsz, conf=conf, verbose=False)
        for p, r in zip(chunk_paths, chunk_results):
            yield p, r
        del chunk_results


@dataclass
class PRData:
    name: str
    recall: np.ndarray
    precision: np.ndarray
    ap: float


def _compute_pr_and_confidences(
    pairs_iter, images_dir: Path, labels_dir: Path, iou_threshold: float = 0.5,
) -> tuple[PRData, list[float]]:
    """Single streaming pass: computes PR-curve detections AND collects confidence
    scores at the same time, so we never need a second inference pass or hold
    all results in memory simultaneously."""
    detections: list[tuple[float, int]] = []
    all_confs: list[float] = []
    total_gt = 0

    for img_path, result in pairs_iter:
        with Image.open(img_path) as pil_img:
            width, height = pil_img.size
        gt_boxes = _yolo_label_boxes(_label_path_for(images_dir, labels_dir, img_path), (height, width))
        total_gt += len(gt_boxes)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_confs = result.boxes.conf.cpu().numpy()
        all_confs.extend(pred_confs.tolist())

        order = np.argsort(-pred_confs)
        matched: set[int] = set()

        for idx in order:
            score = float(pred_confs[idx])
            pred_box = pred_boxes[idx]
            best_iou, best_gt = 0.0, -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched:
                    continue
                iou = _box_iou_xyxy(pred_box, np.asarray(gt_box))
                if iou > best_iou:
                    best_iou, best_gt = iou, gt_idx
            if best_iou >= iou_threshold and best_gt >= 0:
                matched.add(best_gt)
                detections.append((score, 1))
            else:
                detections.append((score, 0))

    if detections and total_gt > 0:
        detections.sort(key=lambda x: x[0], reverse=True)
        labels_arr = np.asarray([lv for _, lv in detections], dtype=int)
        tp = np.cumsum(labels_arr == 1)
        fp = np.cumsum(labels_arr == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(total_gt, 1)
        area_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        ap = float(area_fn(precision, recall)) if len(recall) > 1 else float(precision[-1] * recall[-1])
    else:
        precision, recall, ap = np.array([0.0]), np.array([0.0]), 0.0

    return PRData(name="", recall=recall, precision=precision, ap=ap), all_confs


def _save_confidence_plot(all_confs: list[float], label: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if all_confs:
        ax.hist(all_confs, bins=50, edgecolor="black", alpha=0.7)
    else:
        ax.text(0.5, 0.5, "No predictions found", ha="center", va="center")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution — {label}")
    ax.axvline(x=0.25, color="red", linestyle="--", label="Threshold 0.25")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_pr_plot(pr: PRData, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr.recall, pr.precision, linewidth=2, label=f"{pr.name} (AP={pr.ap:.3f})")
    x_max = max(0.01, float(np.max(pr.recall)))
    y_max = max(0.01, float(np.max(pr.precision)))
    ax.set_xlim(0, min(1.0, x_max * 1.05))
    ax.set_ylim(0, min(1.0, y_max * 1.05))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {pr.name}")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_combined_pr_plot(pr_list: list[PRData], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    gx, gy = 0.0, 0.0
    for pr in pr_list:
        ax.plot(pr.recall, pr.precision, linewidth=2, label=f"{pr.name} (AP={pr.ap:.3f})")
        if len(pr.recall):
            gx = max(gx, float(np.max(pr.recall)))
        if len(pr.precision):
            gy = max(gy, float(np.max(pr.precision)))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Combined Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, min(1.0, max(0.05, gx * 1.05)))
    ax.set_ylim(0, min(1.0, max(0.05, gy * 1.05)))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _metrics_dataframe(results: Any) -> pd.DataFrame:
    if not hasattr(results, "seg"):
        raise RuntimeError("Model did not return segmentation metrics. Use a seg YOLO model.")
    box_precision, box_recall = results.mean_results()[:2]
    rows = [
        ("mAP@50 (mask)", float(results.seg.map50)),
        ("mAP@50-95 (mask)", float(results.seg.map)),
        ("mAP@50 (box)", float(results.box.map50)),
        ("mAP@50-95 (box)", float(results.box.map)),
        ("Precision", float(box_precision)),
        ("Recall", float(box_recall)),
        ("Inference speed (ms/image)", float(results.speed.get("inference", 0.0) or 0.0)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


@dataclass
class EvaluationOutputs:
    metrics_csv: Path
    preview_grid: Path
    table: pd.DataFrame


@dataclass
class ModelResult:
    model: str
    name: str
    outdir: str
    eval: EvaluationOutputs
    predictions: str
    confidence_plot: str
    pr_curve: str
    pr_data: PRData
    ap: float


def _make_temp_yaml_for_val(ds: DatasetInfo, split: str, tmpdir: Path) -> Path:
    """
    Build a minimal yaml for model.val() that points the *requested split*
    at the SAME folder for train/val/test, so YOLO's val() — regardless of
    which split name it internally asks for — always evaluates on your
    actual target folder (fixes the '70 vs 500 images' issue).
    """
    rel_to_root = ds.images_dir.relative_to(ds.root) if ds.images_dir.is_relative_to(ds.root) else ds.images_dir
    yaml_text = (
        f"path: {ds.root.as_posix()}\n"
        f"train: {rel_to_root.as_posix()}\n"
        f"val: {rel_to_root.as_posix()}\n"
        f"test: {rel_to_root.as_posix()}\n"
        f"nc: {ds.nc}\n"
        f"names: {ds.names}\n"
    )
    yaml_path = tmpdir / "val_data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def _run_single_model(
    model_path: Path, name: str, outdir: Path, ds: DatasetInfo,
    imgsz: int, conf: float, iou_threshold: float, batch_size: int,
    max_saved_predictions: int | None, grid_sample_paths: list[Path],
    grid_cache_dir: Path,
    progress: Progress, task_id: Any,
) -> ModelResult:
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    image_paths = _collect_images(ds.images_dir)

    # 1. val() for official metrics — uses a yaml forced to the target split.
    #    workers=0 avoids Ultralytics spawning extra dataloader worker
    #    processes (each holding its own image buffers) — the most likely
    #    cause of a full system crash on a 16GB machine running 9 models
    #    back to back. cache=False stops it writing/reading a shared
    #    labels.cache file that can get corrupted by rapid repeated runs.
    progress.update(task_id, description=f"[cyan]{name}[/] val() on {len(image_paths)} imgs")
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = _make_temp_yaml_for_val(ds, SPLIT, Path(tmpdir))
        val_results = model.val(
            data=str(yaml_path), split="test", imgsz=imgsz,
            batch=batch_size, conf=conf, plots=False, save=False, verbose=False,
            workers=0, cache=False,
        )
    table = _metrics_dataframe(val_results)
    del val_results
    metrics_csv = outdir / "eval" / "evaluation_results.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_csv, index=False)

    # Small preview grid — only 6 images, cheap.
    preview_sample = random.sample(image_paths, min(6, len(image_paths)))
    preview_preds = model.predict(
        source=[str(p) for p in preview_sample], imgsz=imgsz, conf=conf, verbose=False,
    )
    eval_preview = _make_grid(
        [_to_pil(r.plot()) for r in preview_preds],
        outdir / "eval" / "evaluation_preview.png",
    )
    del preview_preds
    eval_out = EvaluationOutputs(metrics_csv=metrics_csv, preview_grid=eval_preview, table=table)

    # 2. SINGLE streaming inference pass over the whole dataset, at low conf
    #    (0.01) so the PR curve sees the full score range. Predictions above
    #    `conf` are saved as annotated images and counted toward the
    #    confidence histogram — all in the same pass, no second forward run.
    progress.update(task_id, description=f"[cyan]{name}[/] inference ({len(image_paths)} imgs)")
    preds_dir = outdir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    write_pool = ThreadPoolExecutor(max_workers=2)  # small pool — keep RAM headroom
    pending = []

    def _record_for_pr():
        """Generator wrapping the streaming predictor, saving display-conf
        annotated images as a side effect while yielding everything to the
        PR/confidence accumulator."""
        nonlocal saved_count
        for img_path, result in _batch_predict_stream(
            model, image_paths, imgsz=imgsz, conf=0.01, batch_size=batch_size,
        ):
            should_save = (
                max_saved_predictions is None or saved_count < max_saved_predictions
            )
            has_high_conf_box = (
                result.boxes is not None and len(result.boxes)
                and float(result.boxes.conf.max()) >= conf
            )
            if should_save and has_high_conf_box:
                annotated = result.plot(conf=True, labels=True, boxes=True, masks=True)
                out_path = preds_dir / img_path.relative_to(ds.images_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pending.append(write_pool.submit(cv2.imwrite, str(out_path), annotated))
                saved_count += 1
            yield img_path, result

    pr, all_confs = _compute_pr_and_confidences(
        _record_for_pr(), ds.images_dir, ds.labels_dir, iou_threshold,
    )
    pr.name = name

    for fut in pending:
        fut.result()
    write_pool.shutdown(wait=True)
    progress.advance(task_id)

    # 3. Plots
    progress.update(task_id, description=f"[cyan]{name}[/] plots")
    conf_path = outdir / "confidence_distribution.png"
    _save_confidence_plot(all_confs, name, conf_path)
    pr_path = outdir / "pr_curve.png"
    _save_pr_plot(pr, pr_path)

    # 5. Run this model on the sample set used for the side-by-side
    #    comparison grid. Chunked by batch_size since this can be the
    #    full dataset (GRID_COMPARISON_SAMPLES = None). Each frame is
    #    written to disk immediately, not held in memory.
    progress.update(task_id, description=f"[cyan]{name}[/] grid samples ({len(grid_sample_paths)})")
    model_cache_dir = grid_cache_dir / name
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    for img_path, result in _batch_predict_stream(
        model, grid_sample_paths, imgsz=imgsz, conf=conf, batch_size=batch_size,
    ):
        frame = result.plot(conf=True, labels=True, boxes=True, masks=True)
        cv2.imwrite(str(model_cache_dir / f"{img_path.stem}.png"), frame)

    # 6. Explicit memory cleanup before moving to the next model.
    #    Critical on low-RAM / CPU-only machines: without this, each model's
    #    weights and any cached tensors stay referenced longer than needed.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    progress.update(task_id, description=f"[green]✓ {name}[/] ({elapsed:.1f}s)")
    progress.advance(task_id)

    return ModelResult(
        model=str(model_path), name=name, outdir=str(outdir),
        eval=eval_out, predictions=str(preds_dir), confidence_plot=str(conf_path),
        pr_curve=str(pr_path), pr_data=pr, ap=pr.ap,
    )


def _print_compare_table(results: list[ModelResult]) -> None:
    table = Table(title="Model Comparison Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("mAP@50 mask", justify="right")
    table.add_column("mAP@50-95 mask", justify="right")
    table.add_column("mAP@50 box", justify="right")
    table.add_column("mAP@50-95 box", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("AP (PR)", justify="right")
    table.add_column("Speed ms", justify="right")

    metric_keys = [
        "mAP@50 (mask)", "mAP@50-95 (mask)", "mAP@50 (box)",
        "mAP@50-95 (box)", "Precision", "Recall", "Inference speed (ms/image)",
    ]

    col_vals: list[list[float]] = [[] for _ in metric_keys]
    ap_vals: list[float] = []
    for r in results:
        for ci, key in enumerate(metric_keys):
            row = r.eval.table[r.eval.table["Metric"] == key]
            col_vals[ci].append(float(row["Value"].iloc[0]) if not row.empty else 0.0)
        ap_vals.append(r.ap)

    best = [
        (min(col_vals[i]) if i == len(metric_keys) - 1 else max(col_vals[i]))
        for i in range(len(metric_keys))
    ]
    best_ap = max(ap_vals)

    for ri, r in enumerate(results):
        cells = []
        for ci in range(len(metric_keys)):
            v = col_vals[ci][ri]
            fmt = f"{v:.1f}" if ci == len(metric_keys) - 1 else f"{v:.4f}"
            cells.append(f"[bold green]{fmt}[/]" if v == best[ci] else fmt)
        ap_fmt = f"{r.ap:.4f}"
        if r.ap == best_ap:
            ap_fmt = f"[bold green]{ap_fmt}[/]"
        table.add_row(r.name, *cells[:6], ap_fmt, cells[6])

    console.print(table)


def main() -> None:
    ds = load_dataset_info(DATA_YAML, SPLIT)
    image_paths = _collect_images(ds.images_dir)

    console.print(Panel(
        f"[bold]Comparing {len(MODELS)} models[/bold]\n"
        f"Data yaml: {ds.yaml_path}\n"
        f"Split:     {SPLIT}  →  {ds.images_dir}\n"
        f"Images:    {len(image_paths)}\n"
        f"Classes:   {ds.names}\n"
        f"Output:    {OUT_ROOT}",
        title="YOLO Paper Comparison", expand=False,
    ))

    if len(image_paths) == 0:
        console.print("[red]No images found — check DATA_YAML / SPLIT in the config block.[/]")
        return

    out_root = _path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Sample images used for the side-by-side "original vs every model"
    # comparison grids. Same images for every model so the grids line up.
    if GRID_COMPARISON_SAMPLES is None:
        grid_sample_paths = list(image_paths)
    else:
        n_grid = min(GRID_COMPARISON_SAMPLES, len(image_paths))
        grid_sample_paths = random.sample(image_paths, n_grid)

    # Each model's annotated grid-sample frames are written to disk as they're
    # produced (NOT kept in RAM) — important when GRID_COMPARISON_SAMPLES is
    # None and this covers the whole dataset across all 9 models.
    grid_cache_dir = out_root / "_grid_frame_cache"
    grid_cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[ModelResult] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30), TaskProgressColumn(), TimeElapsedColumn(),
        console=console, expand=True,
    ) as progress:
        tasks = {
            i: progress.add_task(f"[cyan]{name}[/] waiting...", total=2)
            for i, (_, name) in enumerate(MODELS)
        }
        for i, (model_path, name) in enumerate(MODELS):
            result = _run_single_model(
                model_path=_path(model_path), name=name, outdir=out_root / name,
                ds=ds, imgsz=IMGSZ, conf=CONF, iou_threshold=IOU_THRESHOLD,
                batch_size=BATCH_SIZE, max_saved_predictions=MAX_SAVED_PREDICTIONS,
                grid_sample_paths=grid_sample_paths, grid_cache_dir=grid_cache_dir,
                progress=progress, task_id=tasks[i],
            )
            results.append(result)

    combined_path = out_root / "combined_pr_curve.png"
    _save_combined_pr_plot([r.pr_data for r in results], combined_path)
    console.print(f"[green]Saved combined PR curve →[/] {combined_path}")

    _print_compare_table(results)

    # Build & save the side-by-side comparison grids by reading each model's
    # cached frame back off disk one image at a time — never holds more than
    # one image's worth of frames (9 models) in memory at once.
    if grid_sample_paths:
        grids_dir = out_root / "comparison_grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        model_names = [name for _, name in MODELS]

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30), TaskProgressColumn(), TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Building comparison grids", total=len(grid_sample_paths))
            for p in grid_sample_paths:
                original_bgr = cv2.imread(str(p))
                if original_bgr is None:
                    progress.advance(task)
                    continue
                model_frames = []
                for name in model_names:
                    cached = grid_cache_dir / name / f"{p.stem}.png"
                    if cached.is_file():
                        frame = cv2.imread(str(cached))
                        if frame is not None:
                            model_frames.append((name, frame))
                grid = _make_comparison_grid(original_bgr, model_frames)
                out_path = grids_dir / f"{p.stem}_comparison.png"
                cv2.imwrite(str(out_path), grid)
                progress.advance(task)

        console.print(f"[green]Saved {len(grid_sample_paths)} comparison grids →[/] {grids_dir}")
        shutil.rmtree(grid_cache_dir, ignore_errors=True)

    # Save a single CSV with all models' metrics side by side — handy for the paper
    summary_rows = []
    for r in results:
        row = {"model": r.name, "ap_pr": r.ap}
        for _, m in r.eval.table.iterrows():
            row[m["Metric"]] = m["Value"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "all_models_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    console.print(f"[green]Saved summary CSV →[/] {summary_csv}")


if __name__ == "__main__":
    main()