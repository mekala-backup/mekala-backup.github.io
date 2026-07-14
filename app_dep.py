from __future__ import annotations

import argparse
import os
import random
import shutil
import tempfile
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

# Force CPU mode to prevent system crashes - change to "0" if you have GPU
DEVICE = "cpu"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAME = "transparent_object"


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
class PaperOutputs:
    synth_annotated: Path
    real_annotated: Path
    side_by_side: Path
    synth_confidence_plot: Path
    real_confidence_plot: Path
    synth_pr_curve: Path
    real_pr_curve: Path
    synth_eval: EvaluationOutputs
    real_eval: EvaluationOutputs


@dataclass
class MultiModelPaperOutputs:
    output_root: Path
    model_outputs: list[dict]
    combined_pr_curve: Optional[Path]
    metrics_table: Optional[Path]
    comparison_summary: dict[str, Any]


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _is_dataset_root(root: Path) -> bool:
    return (root / "images").is_dir() and (root / "labels").is_dir()


def _collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
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

    raise FileNotFoundError(f"Expected a YOLO dataset folder containing images/ and labels/: {path}")


def _make_data_yaml(dataset_root: Path) -> str:
    return (
        f"path: {dataset_root.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n"
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )


def _metrics_dataframe(results: Any) -> pd.DataFrame:
    if not hasattr(results, "seg"):
        raise RuntimeError("This model did not return segmentation metrics. Use a segmentation YOLO model.")

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


def _save_metrics_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _to_pil(image_array: Any) -> Image.Image:
    if isinstance(image_array, Image.Image):
        return image_array.convert("RGB")
    return Image.fromarray(image_array[:, :, ::-1]).convert("RGB")


def _make_grid(images: list[Image.Image], path: Path, cols: int = 3) -> Path:
    if not images:
        raise ValueError("No images available to create a preview grid.")

    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    cell_w = min(max(img.width for img in images), 480)
    cell_h = min(max(img.height for img in images), 480)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        fitted = ImageOps.contain(img, (cell_w, cell_h))
        cell = Image.new("RGB", (cell_w, cell_h), "white")
        cell.paste(fitted, ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2))
        canvas.paste(cell, (col * cell_w, row * cell_h))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _image_paths(images_dir: Path) -> list[Path]:
    return _collect_images(images_dir)


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
        xc = float(xc) * width
        yc = float(yc) * height
        w = float(w) * width
        h = float(h) * height
        x1 = max(0.0, xc - w / 2)
        y1 = max(0.0, yc - h / 2)
        x2 = min(float(width), xc + w / 2)
        y2 = min(float(height), yc + h / 2)
        boxes.append([x1, y1, x2, y2])
    return boxes


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _annotate_image(result: Any) -> np.ndarray:
    return result.plot(conf=True, labels=True, boxes=True, masks=True)


def save_predictions(
    model_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> list[Path]:
    """Save predictions with memory-efficient inference."""
    model = YOLO(str(_path(model_path)))
    images_dir = _path(images_dir)
    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = _image_paths(images_dir)
    saved_paths: list[Path] = []
    
    for idx, img_path in enumerate(img_paths):
        try:
            result = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
            annotated = _annotate_image(result)
            out_path = output_dir / img_path.relative_to(images_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)
            saved_paths.append(out_path)
            
            # Memory cleanup every 10 images
            if (idx + 1) % 10 == 0:
                gc.collect()
        except Exception as e:
            print(f"  Warning: Failed to process {img_path.name}: {str(e)}")
            continue

    print(f"Saved {len(saved_paths)} annotated images to {output_dir}")
    del model
    gc.collect()
    return saved_paths


def save_side_by_side(
    synth_model_path: str | Path,
    real_model_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
) -> list[Path]:
    synth_model = YOLO(str(_path(synth_model_path)))
    real_model = YOLO(str(_path(real_model_path)))
    images_dir = _path(images_dir)
    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = _image_paths(images_dir)
    saved_paths: list[Path] = []
    for img_path in img_paths:
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        orig = cv2.resize(orig, (imgsz, imgsz))

        s_res = synth_model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        r_res = real_model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        s_frame = cv2.resize(_annotate_image(s_res), (imgsz, imgsz))
        r_frame = cv2.resize(_annotate_image(r_res), (imgsz, imgsz))

        for frame, label in ((orig, "Original"), (s_frame, "Synth Model"), (r_frame, "Real Model")):
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        combined = cv2.hconcat([orig, s_frame, r_frame])
        out_path = output_dir / img_path.relative_to(images_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), combined)
        saved_paths.append(out_path)

    print(f"Saved {len(saved_paths)} comparison images to {output_dir}")
    return saved_paths


def confidence_distribution(
    model_path: str | Path,
    images_dir: str | Path,
    label: str,
    output_path: str | Path,
    imgsz: int = 640,
    conf: float = 0.01,
    device: str = "cpu",
) -> Path:
    model = YOLO(str(_path(model_path)))
    images_dir = _path(images_dir)
    output_path = _path(output_path)

    all_confs: list[float] = []
    for idx, img_path in enumerate(_image_paths(images_dir)):
        try:
            result = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
            if result.boxes is not None and len(result.boxes):
                all_confs.extend(result.boxes.conf.cpu().numpy().tolist())
        except Exception as e:
            continue
        
        # Memory cleanup
        if (idx + 1) % 20 == 0:
            gc.collect()

    plt.figure(figsize=(8, 4))
    if all_confs:
        plt.hist(all_confs, bins=50, edgecolor="black", alpha=0.7)
    else:
        plt.text(0.5, 0.5, "No predictions found", ha="center", va="center")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title(f"Confidence Distribution — {label}")
    plt.axvline(x=0.25, color="red", linestyle="--", label="Threshold 0.25")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    del model
    gc.collect()
    print(f"Saved confidence distribution to {output_path}")
    return output_path


def plot_pr_curve(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    label: str,
    output_path: str | Path,
    imgsz: int = 640,
    conf: float = 0.01,
    iou_threshold: float = 0.5,
    device: str = "cpu",
) -> Path:
    model = YOLO(str(_path(model_path)))
    images_dir = _path(images_dir)
    labels_dir = _path(labels_dir)
    output_path = _path(output_path)

    detections: list[tuple[float, int]] = []
    total_gt = 0

    for idx, img_path in enumerate(_image_paths(images_dir)):
        with Image.open(img_path) as pil_img:
            width, height = pil_img.size
        gt_boxes = _yolo_label_boxes(_label_path(images_dir, labels_dir, img_path), (height, width))
        total_gt += len(gt_boxes)

        try:
            result = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        except Exception as e:
            continue
        
        if result.boxes is None or len(result.boxes) == 0:
            continue
        
        # Memory cleanup
        if (idx + 1) % 20 == 0:
            gc.collect()

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_confs = result.boxes.conf.cpu().numpy()
        order = np.argsort(-pred_confs)
        matched = set()

        for idx_match in order:
            score = float(pred_confs[idx_match])
            pred_box = pred_boxes[idx_match]
            best_iou = 0.0
            best_gt = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched:
                    continue
                iou = _box_iou_xyxy(pred_box, np.asarray(gt_box))
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            if best_iou >= iou_threshold and best_gt >= 0:
                matched.add(best_gt)
                detections.append((score, 1))
            else:
                detections.append((score, 0))

    plt.figure(figsize=(7, 5))
    if detections and total_gt > 0:
        detections.sort(key=lambda item: item[0], reverse=True)
        labels_arr = np.asarray([label_val for _, label_val in detections], dtype=int)
        tp = np.cumsum(labels_arr == 1)
        fp = np.cumsum(labels_arr == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(total_gt, 1)
        area_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        ap = float(area_fn(precision, recall)) if len(recall) > 1 else float(precision[-1] * recall[-1])
        plt.plot(recall, precision, linewidth=2, label=f"AP={ap:.3f}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, "No detections or labels available", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {label}")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    del model
    gc.collect()
    print(f"Saved PR curve to {output_path}")
    return output_path


def analyze_models(
    synth_model_path: str | Path,
    real_model_path: str | Path,
    test_dataset: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    output_dir: str | Path = "paper_outputs",
    iou_threshold: float = 0.5,
) -> PaperOutputs:
    dataset_root = _find_dataset_root(_path(test_dataset))
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synth_eval = evaluate_model(
        model_path=synth_model_path,
        test_dataset=dataset_root,
        imgsz=imgsz,
        batch=16,
        conf=conf,
        output_dir=output_dir / "synth_eval",
    )
    real_eval = evaluate_model(
        model_path=real_model_path,
        test_dataset=dataset_root,
        imgsz=imgsz,
        batch=16,
        conf=conf,
        output_dir=output_dir / "real_eval",
    )

    synth_annotated_dir = output_dir / "synth_predictions"
    real_annotated_dir = output_dir / "real_predictions"
    comparison_dir = output_dir / "comparison"

    save_predictions(
        synth_model_path,
        images_dir,
        synth_annotated_dir,
        conf=conf,
        imgsz=imgsz,
    )
    save_predictions(
        real_model_path,
        images_dir,
        real_annotated_dir,
        conf=conf,
        imgsz=imgsz,
    )
    save_side_by_side(
        synth_model_path,
        real_model_path,
        images_dir,
        comparison_dir,
        conf=conf,
        imgsz=imgsz,
    )
    synth_conf_plot = confidence_distribution(
        synth_model_path,
        images_dir,
        "Synth Model",
        output_dir / "synth_confidence_distribution.png",
        imgsz=imgsz,
        conf=0.01,
    )
    real_conf_plot = confidence_distribution(
        real_model_path,
        images_dir,
        "Real Model",
        output_dir / "real_confidence_distribution.png",
        imgsz=imgsz,
        conf=0.01,
    )
    synth_pr_curve = plot_pr_curve(
        synth_model_path,
        images_dir,
        labels_dir,
        "Synth Model",
        output_dir / "synth_pr_curve.png",
        imgsz=imgsz,
        conf=0.01,
        iou_threshold=iou_threshold,
    )
    real_pr_curve = plot_pr_curve(
        real_model_path,
        images_dir,
        labels_dir,
        "Real Model",
        output_dir / "real_pr_curve.png",
        imgsz=imgsz,
        conf=0.01,
        iou_threshold=iou_threshold,
    )

    return PaperOutputs(
        synth_annotated=synth_annotated_dir,
        real_annotated=real_annotated_dir,
        side_by_side=comparison_dir,
        synth_confidence_plot=synth_conf_plot,
        real_confidence_plot=real_conf_plot,
        synth_pr_curve=synth_pr_curve,
        real_pr_curve=real_pr_curve,
        synth_eval=synth_eval,
        real_eval=real_eval,
    )


def compare_models(
    models: list[str],
    names: list[str] | None,
    outdirs: list[str] | None,
    test_dataset: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    out_root: str | Path = "evals",
    iou_threshold: float = 0.5,
):
    """Compare N models, save per-model evaluation, predictions, confidence histogram, PR curve, and a combined PR plot.
    Returns list of dicts with per-model paths."""
    models = [Path(m) for m in models]
    test_dataset = _path(test_dataset)
    dataset_root = _find_dataset_root(test_dataset)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    out_root = _path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if names is None:
        names = [m.stem for m in models]
    if outdirs is None:
        outdirs = [str(out_root / n) for n in names]

    if not (len(models) == len(names) == len(outdirs)):
        raise ValueError("--model, --name and --outdir must have the same number of entries if provided")

    pr_data = []
    results = []

    for model_path, name, outdir in zip(models, names, outdirs):
        outdir_p = _path(outdir)
        outdir_p.mkdir(parents=True, exist_ok=True)
        print(f"Evaluating {model_path} -> {outdir_p} (label: {name})")

        eval_out = evaluate_model(
            model_path=model_path,
            test_dataset=dataset_root,
            imgsz=imgsz,
            batch=16,
            conf=conf,
            output_dir=outdir_p / "eval",
        )

        # predictions
        preds_dir = outdir_p / "predictions"
        save_predictions(str(model_path), images_dir, preds_dir, conf=conf, imgsz=imgsz)

        # confidence histogram
        conf_path = outdir_p / "confidence_distribution.png"
        confidence_distribution(str(model_path), images_dir, name, conf_path, imgsz=imgsz, conf=0.01)

        # compute PR arrays (copy of plot_pr_curve logic without plotting to memory)
        model = YOLO(str(model_path))
        detections = []
        total_gt = 0
        for img_path in _image_paths(images_dir):
            with Image.open(img_path) as pil_img:
                width, height = pil_img.size
            gt_boxes = _yolo_label_boxes(_label_path(images_dir, labels_dir, img_path), (height, width))
            total_gt += len(gt_boxes)

            result = model.predict(str(img_path), imgsz=imgsz, conf=0.01, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_confs = result.boxes.conf.cpu().numpy()
            order = np.argsort(-pred_confs)
            matched = set()
            for idx in order:
                score = float(pred_confs[idx])
                pred_box = pred_boxes[idx]
                best_iou = 0.0
                best_gt = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched:
                        continue
                    iou = _box_iou_xyxy(pred_box, np.asarray(gt_box))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx
                if best_iou >= iou_threshold and best_gt >= 0:
                    matched.add(best_gt)
                    detections.append((score, 1))
                else:
                    detections.append((score, 0))

        if detections and total_gt > 0:
            detections.sort(key=lambda item: item[0], reverse=True)
            labels_arr = np.asarray([label_val for _, label_val in detections], dtype=int)
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

        # save per-model PR plot with auto-zoom
        pr_path = outdir_p / "pr_curve.png"
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
        x_max = max(0.01, float(np.max(recall)))
        y_max = max(0.01, float(np.max(precision)))
        plt.xlim(0, min(1.0, x_max * 1.05))
        plt.ylim(0, min(1.0, y_max * 1.05))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {name}")
        plt.legend()
        plt.tight_layout()
        pr_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pr_path)
        plt.close()

        pr_data.append((name, recall, precision, ap))

        results.append({
            "model": str(model_path),
            "name": name,
            "outdir": str(outdir_p),
            "eval": eval_out,
            "predictions": str(preds_dir),
            "confidence_plot": str(conf_path),
            "pr_curve": str(pr_path),
            "ap": ap,
        })

    # combined PR plot
    plt.figure(figsize=(8, 6))
    global_x = 0.0
    global_y = 0.0
    for name, recall, precision, ap in pr_data:
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
        if len(recall):
            global_x = max(global_x, float(np.max(recall)))
        if len(precision):
            global_y = max(global_y, float(np.max(precision)))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curve")
    plt.legend()
    plt.xlim(0, min(1.0, max(0.05, global_x * 1.05)))
    plt.ylim(0, min(1.0, max(0.05, global_y * 1.05)))
    plt.tight_layout()
    combined_path = out_root / "combined_pr_curve.png"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(combined_path)
    plt.close()

    print(f"Saved combined PR curve to {combined_path}")
    return results


def paper_outputs(
    model_paths: list[str | Path],
    test_dataset: str | Path,
    model_names: list[str] | None = None,
    output_root: str | Path = "paper_outputs",
    imgsz: int = 640,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> MultiModelPaperOutputs:
    """Generate paper-ready outputs comparing multiple models (e.g., 9 models).
    
    This creates:
    - Per-model evaluations with metrics
    - Per-model predictions and annotations
    - Confidence distributions for each model
    - Individual PR curves for each model
    - A combined PR curve with all models
    - A metrics comparison table (CSV)
    """
    model_paths = [_path(m) for m in model_paths]
    
    # Validate all model files exist
    print("Validating model files...")
    for model_path in model_paths:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if model_path.suffix.lower() != ".pt":
            raise ValueError(f"Model must be .pt file, got: {model_path.suffix}")
    
    test_dataset = _path(test_dataset)
    output_root = _path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Validate dataset
    print("Validating dataset...")
    dataset_root = _find_dataset_root(test_dataset)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    if model_names is None:
        model_names = [m.stem for m in model_paths]
    
    if len(model_paths) != len(model_names):
        raise ValueError("Number of model paths must match number of model names")
    
    model_outputs = []
    pr_data = []
    metrics_records = []
    
    print(f"Comparing {len(model_paths)} models...")
    
    for idx, (model_path, model_name) in enumerate(zip(model_paths, model_names), 1):
        try:
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{idx}/{len(model_paths)}] Processing {model_name}...")
            
            # Evaluate model
            print(f"  - Evaluating model...")
            eval_output = evaluate_model(
                model_path=model_path,
                test_dataset=dataset_root,
                imgsz=imgsz,
                batch=16,
                conf=conf,
                output_dir=model_dir / "evaluation",
            )
            
            # Save predictions
            print(f"  - Saving predictions...")
            preds_dir = model_dir / "predictions"
            save_predictions(
                str(model_path),
                images_dir,
                preds_dir,
                conf=conf,
                imgsz=imgsz,
                device=DEVICE,
            )
            gc.collect()
            
            # Confidence distribution
            print(f"  - Creating confidence distribution...")
            conf_plot = model_dir / "confidence_distribution.png"
            confidence_distribution(
                str(model_path),
                images_dir,
                model_name,
                conf_plot,
                imgsz=imgsz,
                conf=0.01,
                device=DEVICE,
            )
            gc.collect()
            
            # Compute PR curve
            print(f"  - Computing PR curve...")
            model = YOLO(str(model_path))
            detections = []
            total_gt = 0
            
            for img_path in _image_paths(images_dir):
                with Image.open(img_path) as pil_img:
                    width, height = pil_img.size
                gt_boxes = _yolo_label_boxes(
                    _label_path(images_dir, labels_dir, img_path),
                    (height, width),
                )
                total_gt += len(gt_boxes)
                
                try:
                    result = model.predict(
                        str(img_path),
                        imgsz=imgsz,
                        conf=0.01,
                        verbose=False,
                        device=DEVICE,
                    )[0]
                except Exception as e:
                    continue
                
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                pred_boxes = result.boxes.xyxy.cpu().numpy()
                pred_confs = result.boxes.conf.cpu().numpy()
                order = np.argsort(-pred_confs)
                matched = set()
                
                for idx_box in order:
                    score = float(pred_confs[idx_box])
                    pred_box = pred_boxes[idx_box]
                    best_iou = 0.0
                    best_gt = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched:
                            continue
                        iou = _box_iou_xyxy(pred_box, np.asarray(gt_box))
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt_idx
                    
                    if best_iou >= iou_threshold and best_gt >= 0:
                        matched.add(best_gt)
                        detections.append((score, 1))
                    else:
                        detections.append((score, 0))
            
            # Calculate PR metrics
            if detections and total_gt > 0:
                detections.sort(key=lambda item: item[0], reverse=True)
                labels_arr = np.asarray([label_val for _, label_val in detections], dtype=int)
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
            
            # Save individual PR curve
            print(f"  - Saving PR curve...")
            pr_path = model_dir / "pr_curve.png"
            plt.figure(figsize=(7, 5))
            plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.3f})")
            if len(recall) > 0:
                x_max = max(0.01, float(np.max(recall)))
                y_max = max(0.01, float(np.max(precision)))
                plt.xlim(0, min(1.0, x_max * 1.05))
                plt.ylim(0, min(1.0, y_max * 1.05))
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve — {model_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(pr_path)
            plt.close()
            
            pr_data.append((model_name, recall, precision, ap))
            
            # Extract metrics from eval_output table
            eval_metrics = {}
            for _, row in eval_output.table.iterrows():
                eval_metrics[row["Metric"]] = row["Value"]
            eval_metrics["AP"] = ap
            
            metrics_records.append({
                "Model": model_name,
                **eval_metrics,
            })
            
            model_outputs.append({
                "name": model_name,
                "model_path": str(model_path),
                "output_dir": str(model_dir),
                "evaluation": str(model_dir / "evaluation"),
                "predictions": str(preds_dir),
                "confidence_plot": str(conf_plot),
                "pr_curve": str(pr_path),
                "ap": ap,
                "metrics": eval_metrics,
            })
            
            print(f"  ✓ {model_name} completed (AP={ap:.4f})")
            
            # Cleanup after each model
            del model
            gc.collect()
            
        except Exception as e:
            print(f"\n  ✗ ERROR processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"  - Skipping {model_name} and continuing with next model...")
            gc.collect()
            continue
    
    # Create comparison metrics table
    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_table_path = output_root / "metrics_comparison.csv"
        metrics_df.to_csv(metrics_table_path, index=False)
        print(f"\nSaved metrics comparison table to {metrics_table_path}")
    else:
        metrics_table_path = None
        print(f"\nWarning: No metrics records to save (all models may have failed)")
    
    # Create combined PR curve
    if pr_data:
        plt.figure(figsize=(10, 8))
        global_x = 0.0
        global_y = 0.0
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(pr_data)))
        for (name, recall, precision, ap), color in zip(pr_data, colors):
            plt.plot(
                recall,
                precision,
                linewidth=2.5,
                label=f"{name} (AP={ap:.3f})",
                color=color,
            )
            if len(recall) > 0:
                global_x = max(global_x, float(np.max(recall)))
            if len(precision) > 0:
                global_y = max(global_y, float(np.max(precision)))
        
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"Combined Precision-Recall Curve ({len(pr_data)} Models)", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(1.0, max(0.05, global_x * 1.05)))
        plt.ylim(0, min(1.0, max(0.05, global_y * 1.05)))
        plt.tight_layout()
        
        combined_pr_path = output_root / "combined_pr_curve.png"
        plt.savefig(combined_pr_path, dpi=150)
        plt.close()
        print(f"Saved combined PR curve to {combined_pr_path}")
    else:
        combined_pr_path = None
        print(f"Warning: No PR data to plot (all models may have failed)")
    
    # Create summary
    if model_outputs:
        comparison_summary = {
            "num_models": len(model_outputs),
            "models": [m["name"] for m in model_outputs],
            "test_dataset": str(dataset_root),
            "total_test_images": len(_image_paths(images_dir)),
            "top_model": max(model_outputs, key=lambda x: x["ap"])["name"],
            "top_ap": max(model_outputs, key=lambda x: x["ap"])["ap"],
            "lowest_model": min(model_outputs, key=lambda x: x["ap"])["name"],
            "lowest_ap": min(model_outputs, key=lambda x: x["ap"])["ap"],
            "mean_ap": float(np.mean([m["ap"] for m in model_outputs])),
            "std_ap": float(np.std([m["ap"] for m in model_outputs])),
        }
    else:
        comparison_summary = {
            "num_models": 0,
            "models": [],
            "test_dataset": str(dataset_root),
            "total_test_images": len(_image_paths(images_dir)),
            "top_model": "N/A",
            "top_ap": 0.0,
            "lowest_model": "N/A",
            "lowest_ap": 0.0,
            "mean_ap": 0.0,
            "std_ap": 0.0,
        }
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Models compared: {comparison_summary['num_models']}")
    print(f"Total test images: {comparison_summary['total_test_images']}")
    if comparison_summary['num_models'] > 0:
        print(f"Top model: {comparison_summary['top_model']} (AP={comparison_summary['top_ap']:.4f})")
        print(f"Lowest model: {comparison_summary['lowest_model']} (AP={comparison_summary['lowest_ap']:.4f})")
        print(f"Mean AP: {comparison_summary['mean_ap']:.4f} ± {comparison_summary['std_ap']:.4f}")
    print("="*60)
    
    return MultiModelPaperOutputs(
        output_root=output_root,
        model_outputs=model_outputs,
        combined_pr_curve=combined_pr_path,
        metrics_table=metrics_table_path,
        comparison_summary=comparison_summary,
    )


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
        raise FileNotFoundError(f"No images found in {images_dir}")

    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "data.yaml"
        yaml_path.write_text(_make_data_yaml(dataset_root), encoding="utf-8")

        model = YOLO(str(model_path))
        results = model.val(
            data=str(yaml_path),
            split="test",
            imgsz=int(imgsz),
            batch=int(batch),
            conf=float(conf),
            plots=False,
            save=False,
            verbose=False,
        )

        table = _metrics_dataframe(results)

        sample_paths = random.sample(image_paths, min(6, len(image_paths)))
        predictions = model.predict(
            source=[str(p) for p in sample_paths],
            imgsz=int(imgsz),
            conf=float(conf),
            verbose=False,
        )
        preview_images = [_to_pil(pred.plot()) for pred in predictions]

    metrics_csv = _save_metrics_csv(table, output_dir)
    preview_grid = _make_grid(preview_images, output_dir / "evaluation_preview.png")
    return EvaluationOutputs(metrics_csv=metrics_csv, preview_grid=preview_grid, table=table)


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


def _dataset_preview(output_root: Path) -> list[Image.Image]:
    images = _collect_images(output_root / "images")
    if not images:
        return []
    chosen = random.sample(images, min(6, len(images)))
    preview: list[Image.Image] = []
    for image_path in chosen:
        with Image.open(image_path) as img:
            preview.append(img.convert("RGB").copy())
    return preview


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
        raise ValueError("Total images must be greater than zero.")

    ratio = float(dataset_a_ratio) / 100.0
    n_a = int(total_images * ratio)
    n_b = total_images - n_a

    images_a = _collect_images(dataset_a_root / "images")
    images_b = _collect_images(dataset_b_root / "images")
    if len(images_a) < n_a:
        raise ValueError(f"Dataset A has only {len(images_a)} images, but {n_a} were requested.")
    if len(images_b) < n_b:
        raise ValueError(f"Dataset B has only {len(images_b)} images, but {n_b} were requested.")

    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)
    sampled_a = rng_a.sample(images_a, n_a) if n_a else []
    sampled_b = rng_b.sample(images_b, n_b) if n_b else []

    output_root = _path(output_folder)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)

    copied_a = 0
    copied_b = 0
    skipped: list[str] = []

    for image_path in sampled_a:
        if _copy_sampled_image(image_path, dataset_a_root, output_root):
            copied_a += 1
        else:
            skipped.append(f"Dataset A: {image_path.relative_to(dataset_a_root / 'images')}")

    for image_path in sampled_b:
        if _copy_sampled_image(image_path, dataset_b_root, output_root):
            copied_b += 1
        else:
            skipped.append(f"Dataset B: {image_path.relative_to(dataset_b_root / 'images')}")

    (output_root / "data.yaml").write_text(
        _make_data_yaml(output_root), encoding="utf-8"
    )

    preview_grid = _make_grid(_dataset_preview(output_root), output_root / "mixed_dataset_preview.png")
    summary = {
        "total_created": copied_a + copied_b,
        "dataset_a": copied_a,
        "dataset_b": copied_b,
        "skipped": len(skipped),
        "output_folder": str(output_root),
    }
    return MixOutputs(output_root=output_root, preview_grid=preview_grid, summary=summary, skipped=skipped)


def _print_table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO model evaluator and dataset mixer")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a YOLO segmentation model")
    eval_parser.add_argument("--model", required=True, help="Path to a YOLO .pt model")
    eval_parser.add_argument("--test-data", required=True, help="Path to the test dataset root")
    eval_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    eval_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    eval_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    eval_parser.add_argument("--output-dir", default="evaluation_output", help="Folder for CSV and preview image")

    mix_parser = subparsers.add_parser("mix", help="Create a mixed YOLO dataset")
    mix_parser.add_argument("--dataset-a", required=True, help="Path to Dataset A root")
    mix_parser.add_argument("--dataset-b", required=True, help="Path to Dataset B root")
    mix_parser.add_argument("--total-images", type=int, default=1000, help="Total images in output dataset")
    mix_parser.add_argument("--ratio-a", type=float, default=50.0, help="Percentage of Dataset A in the mix")
    mix_parser.add_argument("--output-folder", default="mixed_dataset", help="Output dataset folder")
    mix_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    paper_parser = subparsers.add_parser("paper", help="Generate paper-ready outputs for two models")
    paper_parser.add_argument("--synth-model", required=True, help="Path to the synthetic model .pt file")
    paper_parser.add_argument("--real-model", required=True, help="Path to the real model .pt file")
    paper_parser.add_argument("--test-data", required=True, help="Path to the test dataset root")
    paper_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    paper_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    paper_parser.add_argument("--output-dir", default="paper_outputs", help="Folder for all paper outputs")
    paper_parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for PR matching")

    # Paper outputs: multi-model comparison
    paper_multi_parser = subparsers.add_parser(
        "paper_outputs",
        help="Generate paper-ready outputs comparing multiple models (e.g., 9 models)",
    )
    paper_multi_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to model .pt files (space-separated)",
    )
    paper_multi_parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Display names for models (optional; defaults to model filename stems)",
    )
    paper_multi_parser.add_argument(
        "--test-data",
        required=True,
        help="Path to the test dataset root",
    )
    paper_multi_parser.add_argument(
        "--output-root",
        default="paper_outputs",
        help="Base output folder for all results",
    )
    paper_multi_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    paper_multi_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    paper_multi_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for PR matching",
    )

    # Compare multiple models
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models and save per-model outputs")
    compare_parser.add_argument("--model", action="append", help="Path to a model .pt file. Repeat for multiple models")
    compare_parser.add_argument("--name", action="append", help="Display name for corresponding model. Repeat or omit to use filename stem")
    compare_parser.add_argument("--outdir", action="append", help="Output directory for corresponding model. Repeat or omit to use <out_root>/<name>")
    compare_parser.add_argument("--test-data", help="Path to the test dataset root")
    compare_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    compare_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    compare_parser.add_argument("--out-root", default="evals", help="Base output folder for compare results")
    compare_parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for PR matching")

    return parser


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def _interactive_main() -> None:
    print("YOLO Evaluation and Dataset Mixer")
    choice = _prompt("Choose mode (evaluate/mix/paper/paper_outputs/q)", "evaluate").lower()
    if choice in {"q", "quit", "exit"}:
        return

    if choice.startswith("e"):
        result = evaluate_model(
            model_path=_prompt("Path to YOLO .pt model"),
            test_dataset=_prompt("Path to test dataset root"),
            imgsz=int(_prompt("Image size", "640")),
            batch=int(_prompt("Batch size", "16")),
            conf=float(_prompt("Confidence threshold", "0.25")),
            output_dir=_prompt("Output folder", "evaluation_output"),
        )
        _print_table(result.table)
        print(f"\nCSV: {result.metrics_csv}")
        print(f"Preview: {result.preview_grid}")
        return

    if choice.startswith("m"):
        result = mix_datasets(
            dataset_a=_prompt("Path to Dataset A root"),
            dataset_b=_prompt("Path to Dataset B root"),
            total_images=int(_prompt("Total images", "1000")),
            dataset_a_ratio=float(_prompt("Dataset A percentage", "50")),
            output_folder=_prompt("Output folder", "mixed_dataset"),
            seed=int(_prompt("Seed", "42")),
        )
        print(
            f"Created {result.summary['total_created']} images "
            f"(A: {result.summary['dataset_a']}, B: {result.summary['dataset_b']})"
        )
        print(f"Skipped: {result.summary['skipped']}")
        print(f"Output: {result.output_root}")
        print(f"Preview: {result.preview_grid}")
        if result.skipped:
            print("Skipped files:")
            for item in result.skipped:
                print(f"  - {item}")
        return

    if choice.startswith("p") and not choice.startswith("pa"):
        result = analyze_models(
            synth_model_path=_prompt("Path to synthetic model .pt"),
            real_model_path=_prompt("Path to real model .pt"),
            test_dataset=_prompt("Path to test dataset root"),
            imgsz=int(_prompt("Image size", "640")),
            conf=float(_prompt("Confidence threshold", "0.25")),
            output_dir=_prompt("Output folder", "paper_outputs"),
            iou_threshold=float(_prompt("PR IoU threshold", "0.5")),
        )
        print(f"Synth annotated: {result.synth_annotated}")
        print(f"Real annotated: {result.real_annotated}")
        print(f"Comparison: {result.side_by_side}")
        print(f"Synth confidence: {result.synth_confidence_plot}")
        print(f"Real confidence: {result.real_confidence_plot}")
        print(f"Synth PR curve: {result.synth_pr_curve}")
        print(f"Real PR curve: {result.real_pr_curve}")
        return

    if choice.startswith("pa"):
        print("\nEnter model paths (one per line). Leave blank line to finish:")
        models = []
        while True:
            model_path = input("Model path: ").strip()
            if not model_path:
                break
            models.append(model_path)
        
        if not models:
            print("No models provided. Aborting.")
            return
        
        print(f"\nEnter display names for {len(models)} models (press Enter to use filename stem):")
        names = []
        for idx, model in enumerate(models, 1):
            name = input(f"Name for model {idx} ({Path(model).stem}): ").strip()
            names.append(name or Path(model).stem)
        
        result = paper_outputs(
            model_paths=models,
            model_names=names,
            test_dataset=_prompt("Path to test dataset root", "/home/mekala/Desktop/evals/real_test"),
            output_root=_prompt("Output folder", "paper_outputs"),
            imgsz=int(_prompt("Image size", "640")),
            conf=float(_prompt("Confidence threshold", "0.25")),
            iou_threshold=float(_prompt("PR IoU threshold", "0.5")),
        )
        
        print("\n" + "="*60)
        print("PAPER OUTPUTS SUMMARY")
        print("="*60)
        print(f"Output root: {result.output_root}")
        print(f"Metrics comparison table: {result.metrics_table}")
        print(f"Combined PR curve: {result.combined_pr_curve}")
        print("\nPer-model outputs:")
        for model_output in result.model_outputs:
            print(f"\n{model_output['name']}:")
            print(f"  - AP: {model_output['ap']:.4f}")
            print(f"  - Predictions: {model_output['predictions']}")
            print(f"  - PR curve: {model_output['pr_curve']}")
        return

    raise ValueError("Unknown mode. Choose evaluate, mix, paper, paper_outputs, or q.")


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
        _print_table(result.table)
        print(f"\nCSV: {result.metrics_csv}")
        print(f"Preview: {result.preview_grid}")
        return

    if args.command == "mix":
        result = mix_datasets(
            dataset_a=args.dataset_a,
            dataset_b=args.dataset_b,
            total_images=args.total_images,
            dataset_a_ratio=args.ratio_a,
            output_folder=args.output_folder,
            seed=args.seed,
        )
        print(
            f"Created {result.summary['total_created']} images "
            f"(A: {result.summary['dataset_a']}, B: {result.summary['dataset_b']})"
        )
        print(f"Skipped: {result.summary['skipped']}")
        print(f"Output: {result.output_root}")
        print(f"Preview: {result.preview_grid}")
        if result.skipped:
            print("Skipped files:")
            for item in result.skipped:
                print(f"  - {item}")
        return

    if args.command == "paper":
        result = analyze_models(
            synth_model_path=args.synth_model,
            real_model_path=args.real_model,
            test_dataset=args.test_data,
            imgsz=args.imgsz,
            conf=args.conf,
            output_dir=args.output_dir,
            iou_threshold=args.iou_threshold,
        )
        print(f"Synth annotated: {result.synth_annotated}")
        print(f"Real annotated: {result.real_annotated}")
        print(f"Comparison: {result.side_by_side}")
        print(f"Synth confidence: {result.synth_confidence_plot}")
        print(f"Real confidence: {result.real_confidence_plot}")
        print(f"Synth PR curve: {result.synth_pr_curve}")
        print(f"Real PR curve: {result.real_pr_curve}")
        return

    if args.command == "paper_outputs":
        model_names = args.names if args.names else None
        result = paper_outputs(
            model_paths=args.models,
            model_names=model_names,
            test_dataset=args.test_data,
            output_root=args.output_root,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou_threshold,
        )
        print("\n" + "="*60)
        print("PAPER OUTPUTS SUMMARY")
        print("="*60)
        print(f"Output root: {result.output_root}")
        print(f"Metrics comparison table: {result.metrics_table}")
        print(f"Combined PR curve: {result.combined_pr_curve}")
        print("\nPer-model outputs:")
        for model_output in result.model_outputs:
            print(f"\n{model_output['name']}:")
            print(f"  - Predictions: {model_output['predictions']}")
            print(f"  - Confidence plot: {model_output['confidence_plot']}")
            print(f"  - PR curve: {model_output['pr_curve']}")
            print(f"  - AP: {model_output['ap']:.4f}")
        return

    if args.command == "compare":
        # Interactive mode: if models not provided as flags, prompt the user
        models = args.model or []
        if not models:
            print("Enter model paths (one per line). Leave blank line to finish:")
            while True:
                v = input("Model path: ").strip()
                if not v:
                    break
                models.append(v)
        if not models:
            print("No models provided. Aborting.")
            return

        # Names
        names = args.name or []
        if len(names) < len(models):
            print("Enter display names for models (press Enter to use filename stem):")
            while len(names) < len(models):
                v = input(f"Name for {models[len(names)]}: ").strip()
                names.append(v or Path(models[len(names)]).stem)

        # Outdirs
        outdirs = args.outdir or []
        out_root_val = args.out_root or "evals"
        if len(outdirs) < len(models):
            print(f"Enter output directories for models (press Enter to use {out_root_val}/<name>):")
            while len(outdirs) < len(models):
                default = str(Path(out_root_val) / names[len(outdirs)])
                v = input(f"Outdir for {names[len(outdirs)]} [{default}]: ").strip()
                outdirs.append(v or default)

        # Test dataset
        test_data = args.test_data or input("Path to test dataset root: ").strip()
        if not test_data:
            print("No test dataset provided. Aborting.")
            return

        results = compare_models(
            models=models,
            names=names,
            outdirs=outdirs,
            test_dataset=test_data,
            imgsz=args.imgsz,
            conf=args.conf,
            out_root=out_root_val,
            iou_threshold=args.iou_threshold,
        )
        print("Compare finished. Per-model outputs:")
        for r in results:
            print(r)
        print(f"Combined PR curve: {Path(out_root_val) / 'combined_pr_curve.png'}")
        return


if __name__ == "__main__":
    main()
