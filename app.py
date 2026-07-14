from __future__ import annotations

import argparse
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

# ── Rich TUI ────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAME = "transparent_object"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class EvaluationOutputs:
    metrics_csv: Path
    preview_grid: Path
    table: pd.DataFrame


@dataclass
class MixOutputs:
    output_root: Path
    preview_grid: Path
    summary: dict[str, Any]
    skipped: list[str]


@dataclass
class PRData:
    name: str
    recall: np.ndarray
    precision: np.ndarray
    ap: float


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


# ── Path / dataset helpers ────────────────────────────────────────────────────

def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _is_dataset_root(root: Path) -> bool:
    return (root / "images").is_dir() and (root / "labels").is_dir()


def _collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_dataset_root(path: Path) -> Path:
    if path.is_file():
        path = path.parent
    if path.name in {"images", "labels"}:
        candidate = path.parent
        if _is_dataset_root(candidate):
            return candidate
    if _is_dataset_root(path):
        return path
    for images_dir in path.rglob("images"):
        if not images_dir.is_dir():
            continue
        candidate = images_dir.parent
        if _is_dataset_root(candidate):
            return candidate
    raise FileNotFoundError(f"No YOLO dataset (images/ + labels/) found under: {path}")


def _make_data_yaml(dataset_root: Path) -> str:
    return (
        f"path: {dataset_root.as_posix()}\n"
        f"train: images\nval: images\ntest: images\n"
        f"nc: 1\nnames: ['{CLASS_NAME}']\n"
    )


def _label_path(images_dir: Path, labels_dir: Path, image_path: Path) -> Path:
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
        xc, yc, w, h = float(xc)*width, float(yc)*height, float(w)*width, float(h)*height
        boxes.append([max(0.0, xc-w/2), max(0.0, yc-h/2),
                       min(float(width), xc+w/2), min(float(height), yc+h/2)])
    return boxes


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
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
        cell.paste(fitted, ((cell_w-fitted.width)//2, (cell_h-fitted.height)//2))
        canvas.paste(cell, (col*cell_w, row*cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# ── Core inference — batch all images in one pass ────────────────────────────

def _batch_predict(model: YOLO, image_paths: list[Path], imgsz: int, conf: float,
                   batch_size: int = 32) -> list[Any]:
    """Run predict in batches; returns results in same order as image_paths."""
    all_results = []
    paths_str = [str(p) for p in image_paths]
    for i in range(0, len(paths_str), batch_size):
        chunk = paths_str[i:i+batch_size]
        all_results.extend(model.predict(source=chunk, imgsz=imgsz, conf=conf, verbose=False))
    return all_results


def _compute_pr_from_results(
    results: list[Any],
    image_paths: list[Path],
    images_dir: Path,
    labels_dir: Path,
    iou_threshold: float = 0.5,
) -> PRData:
    """Compute PR curve from already-computed results (no re-inference)."""
    detections: list[tuple[float, int]] = []
    total_gt = 0

    for img_path, result in zip(image_paths, results):
        with Image.open(img_path) as pil_img:
            width, height = pil_img.size
        gt_boxes = _yolo_label_boxes(_label_path(images_dir, labels_dir, img_path), (height, width))
        total_gt += len(gt_boxes)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_confs = result.boxes.conf.cpu().numpy()
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
        precision = np.array([0.0])
        recall = np.array([0.0])
        ap = 0.0

    return PRData(name="", recall=recall, precision=precision, ap=ap)


# ── Plotting helpers (fast, no interactive display) ──────────────────────────

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


# ── Metrics ──────────────────────────────────────────────────────────────────

def _metrics_dataframe(results: Any) -> pd.DataFrame:
    if not hasattr(results, "seg"):
        raise RuntimeError("Model did not return segmentation metrics. Use a seg YOLO model.")
    box_precision, box_recall = results.mean_results()[:2]
    rows = [
        ("mAP@50 (mask)",          float(results.seg.map50)),
        ("mAP@50-95 (mask)",       float(results.seg.map)),
        ("mAP@50 (box)",           float(results.box.map50)),
        ("mAP@50-95 (box)",        float(results.box.map)),
        ("Precision",              float(box_precision)),
        ("Recall",                 float(box_recall)),
        ("Inference speed (ms/image)", float(results.speed.get("inference", 0.0) or 0.0)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ── Single model evaluation ───────────────────────────────────────────────────

def evaluate_model(
    model_path: str | Path,
    test_dataset: str | Path,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.25,
    output_dir: str | Path = "evaluation_output",
) -> EvaluationOutputs:
    model_path = _path(model_path)
    if not model_path.is_file() or model_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Invalid YOLO model: {model_path}")

    dataset_root = _find_dataset_root(_path(test_dataset))
    images_dir = dataset_root / "images"
    image_paths = _collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images in {images_dir}")

    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "data.yaml"
        yaml_path.write_text(_make_data_yaml(dataset_root), encoding="utf-8")

        model = YOLO(str(model_path))
        val_results = model.val(
            data=str(yaml_path), split="test", imgsz=int(imgsz),
            batch=int(batch), conf=float(conf), plots=False, save=False, verbose=False,
        )
        table = _metrics_dataframe(val_results)

        sample_paths = random.sample(image_paths, min(6, len(image_paths)))
        predictions = model.predict(
            source=[str(p) for p in sample_paths],
            imgsz=int(imgsz), conf=float(conf), verbose=False,
        )
        preview_images = [_to_pil(pred.plot()) for pred in predictions]

    metrics_csv = output_dir / "evaluation_results.csv"
    table.to_csv(metrics_csv, index=False)
    preview_grid = _make_grid(preview_images, output_dir / "evaluation_preview.png")
    return EvaluationOutputs(metrics_csv=metrics_csv, preview_grid=preview_grid, table=table)


# ── Dataset mix ───────────────────────────────────────────────────────────────

def _remap_label(src_label: Path, dst_label: Path) -> None:
    lines: list[str] = []
    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        lines.append(f"0 {parts[1]}" if len(parts) > 1 else "0")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_sampled_image(image_path: Path, source_root: Path, output_root: Path) -> bool:
    rel_path = image_path.relative_to(source_root / "images")
    label_src = source_root / "labels" / rel_path.with_suffix(".txt")
    if not label_src.is_file():
        return False
    image_dst = output_root / "images" / rel_path
    label_dst = output_root / "labels" / rel_path.with_suffix(".txt")
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dst)
    _remap_label(label_src, label_dst)
    return True


def mix_datasets(
    dataset_a: str | Path,
    dataset_b: str | Path,
    total_images: int = 1000,
    dataset_a_ratio: float = 50.0,
    output_folder: str | Path = "mixed_dataset",
    seed: int = 42,
) -> MixOutputs:
    dataset_a_root = _find_dataset_root(_path(dataset_a))
    dataset_b_root = _find_dataset_root(_path(dataset_b))

    total_images = int(total_images)
    if total_images <= 0:
        raise ValueError("Total images must be > 0.")

    ratio = float(dataset_a_ratio) / 100.0
    n_a = int(total_images * ratio)
    n_b = total_images - n_a

    images_a = _collect_images(dataset_a_root / "images")
    images_b = _collect_images(dataset_b_root / "images")

    if len(images_a) < n_a:
        raise ValueError(f"Dataset A has {len(images_a)} images, requested {n_a}.")
    if len(images_b) < n_b:
        raise ValueError(f"Dataset B has {len(images_b)} images, requested {n_b}.")

    sampled_a = random.Random(seed).sample(images_a, n_a) if n_a else []
    sampled_b = random.Random(seed + 1).sample(images_b, n_b) if n_b else []

    output_root = _path(output_folder)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)

    copied_a, copied_b, skipped = 0, 0, []

    def _copy(args: tuple[Path, Path, Path]) -> tuple[bool, str]:
        img, src_root, dst_root = args
        ok = _copy_sampled_image(img, src_root, dst_root)
        label = "Dataset A" if src_root == dataset_a_root else "Dataset B"
        return ok, f"{label}: {img.relative_to(src_root / 'images')}"

    tasks_a = [(img, dataset_a_root, output_root) for img in sampled_a]
    tasks_b = [(img, dataset_b_root, output_root) for img in sampled_b]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Copying images...", total=len(tasks_a) + len(tasks_b))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for ok, label in pool.map(_copy, tasks_a + tasks_b):
                if ok:
                    if "Dataset A" in label:
                        copied_a += 1
                    else:
                        copied_b += 1
                else:
                    skipped.append(label)
                progress.advance(task)

    (output_root / "data.yaml").write_text(_make_data_yaml(output_root), encoding="utf-8")

    chosen = random.sample(_collect_images(output_root / "images"), min(6, copied_a + copied_b))
    preview_images = [Image.open(p).convert("RGB") for p in chosen]
    preview_grid = _make_grid(preview_images, output_root / "mixed_dataset_preview.png")

    summary = {
        "total_created": copied_a + copied_b,
        "dataset_a": copied_a,
        "dataset_b": copied_b,
        "skipped": len(skipped),
        "output_folder": str(output_root),
    }
    return MixOutputs(output_root=output_root, preview_grid=preview_grid, summary=summary, skipped=skipped)


# ── Compare N models (optimised: single inference pass per model) ────────────

def _run_single_model(
    model_path: Path,
    name: str,
    outdir: Path,
    images_dir: Path,
    labels_dir: Path,
    dataset_root: Path,
    imgsz: int,
    conf: float,
    iou_threshold: float,
    progress: Progress,
    task_id: Any,
) -> ModelResult:
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    image_paths = _collect_images(images_dir)

    # ── 1. val() for official metrics ────────────────────────────────────────
    progress.update(task_id, description=f"[cyan]{name}[/] val()")
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "data.yaml"
        yaml_path.write_text(_make_data_yaml(dataset_root), encoding="utf-8")
        val_results = model.val(
            data=str(yaml_path), split="test", imgsz=imgsz,
            batch=16, conf=conf, plots=False, save=False, verbose=False,
        )
    table = _metrics_dataframe(val_results)
    metrics_csv = outdir / "eval" / "evaluation_results.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_csv, index=False)
    eval_preview = _make_grid(
        [_to_pil(r.plot()) for r in model.predict(
            source=[str(p) for p in random.sample(image_paths, min(6, len(image_paths)))],
            imgsz=imgsz, conf=conf, verbose=False,
        )],
        outdir / "eval" / "evaluation_preview.png",
    )
    eval_out = EvaluationOutputs(metrics_csv=metrics_csv, preview_grid=eval_preview, table=table)

    # ── 2. Single full-dataset inference pass (low conf for PR) ─────────────
    progress.update(task_id, description=f"[cyan]{name}[/] inference ({len(image_paths)} imgs)")
    all_results = _batch_predict(model, image_paths, imgsz=imgsz, conf=0.01, batch_size=32)
    progress.advance(task_id)

    # ── 3. Save annotated predictions (parallelised per-image write) ─────────
    progress.update(task_id, description=f"[cyan]{name}[/] saving predictions")
    preds_dir = outdir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Re-run at display conf for annotated images only (use batch predict)
    display_results = _batch_predict(model, image_paths, imgsz=imgsz, conf=conf, batch_size=32)

    def _write_annotated(args: tuple[Path, Any]) -> None:
        img_path, result = args
        annotated = result.plot(conf=True, labels=True, boxes=True, masks=True)
        out_path = preds_dir / img_path.relative_to(images_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_write_annotated, zip(image_paths, display_results)))

    # ── 4. Confidence distribution (from low-conf results) ──────────────────
    progress.update(task_id, description=f"[cyan]{name}[/] plots")
    all_confs = []
    for result in all_results:
        if result.boxes is not None and len(result.boxes):
            all_confs.extend(result.boxes.conf.cpu().numpy().tolist())
    conf_path = outdir / "confidence_distribution.png"
    _save_confidence_plot(all_confs, name, conf_path)

    # ── 5. PR curve (reuse all_results — no re-inference) ───────────────────
    pr = _compute_pr_from_results(all_results, image_paths, images_dir, labels_dir, iou_threshold)
    pr.name = name
    pr_path = outdir / "pr_curve.png"
    _save_pr_plot(pr, pr_path)

    elapsed = time.time() - t0
    progress.update(task_id, description=f"[green]✓ {name}[/] ({elapsed:.1f}s)")
    progress.advance(task_id)

    return ModelResult(
        model=str(model_path),
        name=name,
        outdir=str(outdir),
        eval=eval_out,
        predictions=str(preds_dir),
        confidence_plot=str(conf_path),
        pr_curve=str(pr_path),
        pr_data=pr,
        ap=pr.ap,
    )


def compare_models(
    models: list[str | Path],
    names: list[str] | None,
    outdirs: list[str] | None,
    test_dataset: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    out_root: str | Path = "evals",
    iou_threshold: float = 0.5,
) -> list[ModelResult]:
    models = [_path(m) for m in models]
    dataset_root = _find_dataset_root(_path(test_dataset))
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    out_root = _path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if names is None:
        names = [m.stem for m in models]
    if outdirs is None:
        outdirs = [str(out_root / n) for n in names]

    if not (len(models) == len(names) == len(outdirs)):
        raise ValueError("--model, --name, --outdir must have equal counts.")

    console.print(Panel(
        f"[bold]Comparing {len(models)} models[/bold]\n"
        f"Dataset: {dataset_root}\n"
        f"Images:  {len(_collect_images(images_dir))}\n"
        f"Output:  {out_root}",
        title="YOLO Compare", expand=False
    ))

    results: list[ModelResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        # One progress task per model (2 ticks each: inference done, plots done)
        tasks = {
            i: progress.add_task(f"[cyan]{names[i]}[/] waiting...", total=2)
            for i in range(len(models))
        }

        # Models run sequentially (GPU is the bottleneck; parallel would OOM)
        for i, (model_path, name, outdir) in enumerate(zip(models, names, outdirs)):
            result = _run_single_model(
                model_path=model_path,
                name=name,
                outdir=_path(outdir),
                images_dir=images_dir,
                labels_dir=labels_dir,
                dataset_root=dataset_root,
                imgsz=imgsz,
                conf=conf,
                iou_threshold=iou_threshold,
                progress=progress,
                task_id=tasks[i],
            )
            results.append(result)

    # ── Combined PR plot ─────────────────────────────────────────────────────
    combined_path = out_root / "combined_pr_curve.png"
    _save_combined_pr_plot([r.pr_data for r in results], combined_path)
    console.print(f"[green]Saved combined PR curve →[/] {combined_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    _print_compare_table(results)

    return results


def _print_compare_table(results: list[ModelResult]) -> None:
    table = Table(title="Model Comparison Summary", box=box.SIMPLE_HEAVY, show_lines=False)
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

    # Find best values per column for highlighting
    col_vals: list[list[float]] = [[] for _ in metric_keys]
    ap_vals: list[float] = []
    for r in results:
        for ci, key in enumerate(metric_keys):
            row = r.eval.table[r.eval.table["Metric"] == key]
            col_vals[ci].append(float(row["Value"].iloc[0]) if not row.empty else 0.0)
        ap_vals.append(r.ap)

    # For speed: lower is better; for everything else: higher is better
    best = [
        (min(col_vals[i]) if i == len(metric_keys) - 1 else max(col_vals[i]))
        for i in range(len(metric_keys))
    ]
    best_ap = max(ap_vals)

    for ri, r in enumerate(results):
        cells = []
        for ci, key in enumerate(metric_keys):
            v = col_vals[ci][ri]
            fmt = f"{v:.1f}" if ci == len(metric_keys)-1 else f"{v:.4f}"
            cells.append(f"[bold green]{fmt}[/]" if v == best[ci] else fmt)

        ap_fmt = f"{r.ap:.4f}"
        if r.ap == best_ap:
            ap_fmt = f"[bold green]{ap_fmt}[/]"

        table.add_row(r.name, *cells[:6], ap_fmt, cells[6])

    console.print(table)


# ── Paper outputs (thin wrapper around compare) ───────────────────────────────

def analyze_models(
    synth_model_path: str | Path,
    real_model_path: str | Path,
    test_dataset: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    output_dir: str | Path = "paper_outputs",
    iou_threshold: float = 0.5,
) -> dict:
    output_dir = _path(output_dir)
    results = compare_models(
        models=[synth_model_path, real_model_path],
        names=["synth", "real"],
        outdirs=[str(output_dir / "synth"), str(output_dir / "real")],
        test_dataset=test_dataset,
        imgsz=imgsz,
        conf=conf,
        out_root=output_dir,
        iou_threshold=iou_threshold,
    )
    # Side-by-side comparison images
    synth_r, real_r = results[0], results[1]
    dataset_root = _find_dataset_root(_path(test_dataset))
    images_dir = dataset_root / "images"
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    synth_model = YOLO(str(_path(synth_model_path)))
    real_model  = YOLO(str(_path(real_model_path)))

    img_paths = _collect_images(images_dir)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Side-by-side comparison images...", total=len(img_paths))
        s_results = _batch_predict(synth_model, img_paths, imgsz, conf)
        r_results = _batch_predict(real_model,  img_paths, imgsz, conf)

        def _write_sbs(args: tuple[Path, Any, Any]) -> None:
            img_path, s_res, r_res = args
            orig = cv2.imread(str(img_path))
            if orig is None:
                return
            orig = cv2.resize(orig, (imgsz, imgsz))
            s_frame = cv2.resize(s_res.plot(), (imgsz, imgsz))
            r_frame = cv2.resize(r_res.plot(), (imgsz, imgsz))
            for frame, lbl in ((orig, "Original"), (s_frame, "Synth"), (r_frame, "Real")):
                cv2.putText(frame, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            out_path = comparison_dir / img_path.relative_to(images_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.hconcat([orig, s_frame, r_frame]))

        with ThreadPoolExecutor(max_workers=6) as pool:
            for _ in pool.map(_write_sbs, zip(img_paths, s_results, r_results)):
                progress.advance(task)

    return {
        "synth": synth_r,
        "real": real_r,
        "comparison_dir": comparison_dir,
        "combined_pr": output_dir / "combined_pr_curve.png",
    }


# ── TUI helpers ───────────────────────────────────────────────────────────────

def _print_eval_table(result: EvaluationOutputs, name: str = "") -> None:
    title = f"Evaluation Results{f' — {name}' if name else ''}"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for _, row in result.table.iterrows():
        table.add_row(str(row["Metric"]), f"{float(row['Value']):.4f}")
    console.print(table)


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"  {text}{suffix}: ").strip()
    return value or (default or "")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO segmentation evaluator, dataset mixer, and multi-model comparator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # evaluate
    ep = sub.add_parser("evaluate", help="Evaluate a single YOLO segmentation model")
    ep.add_argument("--model",      required=True)
    ep.add_argument("--test-data",  required=True)
    ep.add_argument("--imgsz",      type=int,   default=640)
    ep.add_argument("--batch",      type=int,   default=16)
    ep.add_argument("--conf",       type=float, default=0.25)
    ep.add_argument("--output-dir", default="evaluation_output")

    # mix
    mp = sub.add_parser("mix", help="Create a mixed YOLO dataset from two sources")
    mp.add_argument("--dataset-a",    required=True)
    mp.add_argument("--dataset-b",    required=True)
    mp.add_argument("--total-images", type=int,   default=1000)
    mp.add_argument("--ratio-a",      type=float, default=50.0)
    mp.add_argument("--output-folder", default="mixed_dataset")
    mp.add_argument("--seed",         type=int,   default=42)

    # paper
    pp = sub.add_parser("paper", help="Generate paper figures for synth vs real models")
    pp.add_argument("--synth-model", required=True)
    pp.add_argument("--real-model",  required=True)
    pp.add_argument("--test-data",   required=True)
    pp.add_argument("--imgsz",       type=int,   default=640)
    pp.add_argument("--conf",        type=float, default=0.25)
    pp.add_argument("--output-dir",  default="paper_outputs")
    pp.add_argument("--iou-threshold", type=float, default=0.5)

    # compare
    cp = sub.add_parser("compare", help="Compare N models with full per-model outputs")
    cp.add_argument("--model",  action="append", metavar="MODEL",
                    help="Path to .pt file. Repeat for each model.")
    cp.add_argument("--name",   action="append", metavar="NAME",
                    help="Display name. Repeat to match --model order.")
    cp.add_argument("--outdir", action="append", metavar="DIR",
                    help="Output dir per model. Repeat or omit.")
    cp.add_argument("--test-data",     required=True)
    cp.add_argument("--imgsz",         type=int,   default=640)
    cp.add_argument("--conf",          type=float, default=0.25)
    cp.add_argument("--out-root",      default="evals")
    cp.add_argument("--iou-threshold", type=float, default=0.5)

    return parser


def _interactive_main() -> None:
    console.print(Panel(
        "[bold]YOLO Evaluation & Dataset Mixer[/bold]\n"
        "evaluate · mix · paper · compare",
        expand=False,
    ))
    choice = _prompt("Mode", "evaluate").lower()
    if choice in {"q", "quit", "exit"}:
        return

    if choice.startswith("ev"):
        result = evaluate_model(
            model_path=_prompt("Model .pt path"),
            test_dataset=_prompt("Test dataset root"),
            imgsz=int(_prompt("Image size", "640")),
            batch=int(_prompt("Batch size", "16")),
            conf=float(_prompt("Confidence threshold", "0.25")),
            output_dir=_prompt("Output folder", "evaluation_output"),
        )
        _print_eval_table(result)
        console.print(f"CSV:     {result.metrics_csv}")
        console.print(f"Preview: {result.preview_grid}")

    elif choice.startswith("m"):
        result = mix_datasets(
            dataset_a=_prompt("Dataset A root"),
            dataset_b=_prompt("Dataset B root"),
            total_images=int(_prompt("Total images", "1000")),
            dataset_a_ratio=float(_prompt("Dataset A %", "50")),
            output_folder=_prompt("Output folder", "mixed_dataset"),
            seed=int(_prompt("Seed", "42")),
        )
        console.print(
            f"Created {result.summary['total_created']} images "
            f"(A: {result.summary['dataset_a']}, B: {result.summary['dataset_b']})"
        )
        console.print(f"Output:  {result.output_root}")
        console.print(f"Preview: {result.preview_grid}")

    elif choice.startswith("p"):
        out = analyze_models(
            synth_model_path=_prompt("Synth model .pt"),
            real_model_path=_prompt("Real model .pt"),
            test_dataset=_prompt("Test dataset root"),
            imgsz=int(_prompt("Image size", "640")),
            conf=float(_prompt("Confidence threshold", "0.25")),
            output_dir=_prompt("Output folder", "paper_outputs"),
            iou_threshold=float(_prompt("PR IoU threshold", "0.5")),
        )
        console.print(f"Comparison dir: {out['comparison_dir']}")
        console.print(f"Combined PR:    {out['combined_pr']}")

    elif choice.startswith("c"):
        models_input: list[str] = []
        console.print("Enter model paths (blank line to finish):")
        while True:
            v = input("  model path: ").strip()
            if not v:
                break
            models_input.append(v)
        if not models_input:
            console.print("[red]No models provided.[/]")
            return

        names_input: list[str] = []
        console.print("Enter display names (Enter to use filename stem):")
        for mp in models_input:
            v = input(f"  name for {Path(mp).name}: ").strip()
            names_input.append(v or Path(mp).stem)

        test_data = _prompt("Test dataset root")
        out_root  = _prompt("Output root", "evals")

        results = compare_models(
            models=models_input,
            names=names_input,
            outdirs=None,
            test_dataset=test_data,
            imgsz=int(_prompt("Image size", "640")),
            conf=float(_prompt("Confidence threshold", "0.25")),
            out_root=out_root,
            iou_threshold=float(_prompt("IoU threshold", "0.5")),
        )
        console.print(f"\n[green]Done.[/] Combined PR: {_path(out_root) / 'combined_pr_curve.png'}")

    else:
        console.print(f"[red]Unknown mode:[/] {choice}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        _interactive_main()
        return

    if args.command == "evaluate":
        result = evaluate_model(
            model_path=args.model,
            test_dataset=args.test_data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            output_dir=args.output_dir,
        )
        _print_eval_table(result)
        console.print(f"CSV:     {result.metrics_csv}")
        console.print(f"Preview: {result.preview_grid}")

    elif args.command == "mix":
        result = mix_datasets(
            dataset_a=args.dataset_a,
            dataset_b=args.dataset_b,
            total_images=args.total_images,
            dataset_a_ratio=args.ratio_a,
            output_folder=args.output_folder,
            seed=args.seed,
        )
        console.print(
            f"Created {result.summary['total_created']} images "
            f"(A: {result.summary['dataset_a']}, B: {result.summary['dataset_b']})\n"
            f"Output:  {result.output_root}\nPreview: {result.preview_grid}"
        )
        if result.skipped:
            console.print(f"[yellow]Skipped {len(result.skipped)} images[/]")

    elif args.command == "paper":
        out = analyze_models(
            synth_model_path=args.synth_model,
            real_model_path=args.real_model,
            test_dataset=args.test_data,
            imgsz=args.imgsz,
            conf=args.conf,
            output_dir=args.output_dir,
            iou_threshold=args.iou_threshold,
        )
        console.print(f"Comparison: {out['comparison_dir']}")
        console.print(f"Combined PR: {out['combined_pr']}")

    elif args.command == "compare":
        models = args.model or []
        if not models:
            console.print("[red]Provide at least one --model flag.[/]")
            return
        results = compare_models(
            models=models,
            names=args.name,
            outdirs=args.outdir,
            test_dataset=args.test_data,
            imgsz=args.imgsz,
            conf=args.conf,
            out_root=args.out_root,
            iou_threshold=args.iou_threshold,
        )
        console.print(f"\n[green]Done.[/] {len(results)} models evaluated.")
        console.print(f"Combined PR → {_path(args.out_root) / 'combined_pr_curve.png'}")


if __name__ == "__main__":
    main()

