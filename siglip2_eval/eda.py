#!/usr/bin/env python3
"""
eda.py -- Exploratory Data Analysis for hkv_subsample_p2 (HyperKvasir subsample)
================================================================================

Goal
----
Before stress-testing the SigLIP2 encoder, we need a clear picture of
*what is in the dataset*. This script walks the three split folders (
``hkv_subsample_p2-...-3-00{1,2,3}``), merges them into a single logical
dataset, and produces:

1. **Inventory tables** (CSV) -- per-class image counts, per-tract /
   per-category breakdowns, video counts, segmentation-mask counts.
2. **Imbalance diagnostics** -- effective number of samples (Cui et al.,
   2019), Shannon entropy, Gini index, head/tail ratio.
3. **Image statistics** -- resolution distribution, aspect ratios, mean
   per-channel intensity, brightness, saturation, specular-highlight
   fraction (proxy for instrument glare). Aggregated and per-class.
4. **Duplicate / near-duplicate detection** -- perceptual-hash (pHash) on
   a downsampled image and Hamming-distance clusters.
5. **Video metadata** -- duration, fps, frame count, resolution.
6. **Segmentation summary** -- mask-to-image area ratio, bbox count per
   image, label distribution.
7. **Plots** (PNG) saved alongside the tables.

All outputs land in ``--out`` (default ``./results/eda``).

Why this script matters for the SigLIP2 evaluation
--------------------------------------------------
* The class imbalance is severe (1 - 256 samples). Any zero-shot or
  linear-probe accuracy MUST be reported with macro-F1 / balanced
  accuracy alongside top-1, otherwise majority classes dominate.
* Resolution variance motivates the resolution sweep in ``benchmark.py``.
* Specular-highlight fraction motivates the corruption robustness suite.
* Duplicate detection prevents train/test leakage in the linear probe.
* Video metadata feeds the temporal-coherence experiment.

Usage
-----
    python eda.py --data /path/to/toyDataset --out ./results/eda
    python eda.py --data /path/to/toyDataset --out ./results/eda --quick

``--quick`` skips per-image color statistics and dedup, useful for a
first pass on a slow disk.

Dependencies
------------
    pip install pillow numpy pandas matplotlib opencv-python imagehash tqdm

The script is CPU-only and does not require a GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib is required: pip install matplotlib") from e

# Optional dependencies -- gracefully degrade if missing.
try:
    import imagehash  # type: ignore
    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXT = {".avi", ".mp4", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Dataset discovery -- merges the 3 split folders into one logical view.
# ---------------------------------------------------------------------------
@dataclass
class ImageRecord:
    path: str
    tract: str           # lower-gi-tract / upper-gi-tract
    category: str        # anatomical-landmarks / pathological-findings / ...
    class_name: str      # raw folder name, e.g. "bbps-2-3"
    split: str           # "001" / "002" / "003" -- archive of origin


@dataclass
class VideoRecord:
    path: str
    tract: str
    category: str
    class_name: str
    split: str


def find_split_roots(data_dir: str) -> List[Tuple[str, str]]:
    """Return ``[(split_id, hkv_subsample_p2_root), ...]`` for each archive."""
    out: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        # Expected folder name: "hkv_subsample_p2-<timestamp>-3-00X"
        if "hkv_subsample_p2" not in entry:
            continue
        split = entry.rsplit("-", 1)[-1]  # "001" etc.
        inner = os.path.join(full, "hkv_subsample_p2")
        if os.path.isdir(inner):
            out.append((split, inner))
    if not out:
        # Fallback: maybe user pointed --data directly at hkv_subsample_p2
        if os.path.basename(data_dir.rstrip("/")) == "hkv_subsample_p2":
            out.append(("000", data_dir))
        else:
            inner = os.path.join(data_dir, "hkv_subsample_p2")
            if os.path.isdir(inner):
                out.append(("000", inner))
    return out


def discover_images(roots: List[Tuple[str, str]]) -> List[ImageRecord]:
    """Walk ``labeled-images/<tract>/<category>/<class>/*`` across all splits."""
    records: List[ImageRecord] = []
    for split, root in roots:
        labeled = os.path.join(root, "labeled-images")
        if not os.path.isdir(labeled):
            continue
        for tract in sorted(os.listdir(labeled)):
            t_dir = os.path.join(labeled, tract)
            if not os.path.isdir(t_dir):
                continue
            for category in sorted(os.listdir(t_dir)):
                c_dir = os.path.join(t_dir, category)
                if not os.path.isdir(c_dir):
                    continue
                for cls in sorted(os.listdir(c_dir)):
                    cls_dir = os.path.join(c_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for fname in sorted(os.listdir(cls_dir)):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in IMG_EXT:
                            records.append(ImageRecord(
                                path=os.path.join(cls_dir, fname),
                                tract=tract, category=category,
                                class_name=cls, split=split,
                            ))
    return records


def discover_videos(roots: List[Tuple[str, str]]) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    for split, root in roots:
        labeled = os.path.join(root, "labeled-videos")
        if not os.path.isdir(labeled):
            continue
        for tract in sorted(os.listdir(labeled)):
            t_dir = os.path.join(labeled, tract)
            if not os.path.isdir(t_dir):
                continue
            for category in sorted(os.listdir(t_dir)):
                c_dir = os.path.join(t_dir, category)
                if not os.path.isdir(c_dir):
                    continue
                for cls in sorted(os.listdir(c_dir)):
                    cls_dir = os.path.join(c_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for fname in sorted(os.listdir(cls_dir)):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in VID_EXT:
                            records.append(VideoRecord(
                                path=os.path.join(cls_dir, fname),
                                tract=tract, category=category,
                                class_name=cls, split=split,
                            ))
    return records


def discover_segmented(roots: List[Tuple[str, str]]):
    """Collect segmentation images, masks, and bbox metadata across splits."""
    images: List[str] = []
    masks: List[str] = []
    bbox_meta: dict = {}
    for split, root in roots:
        seg = os.path.join(root, "segmented-images")
        if not os.path.isdir(seg):
            continue
        # images
        img_dir = os.path.join(seg, "images")
        if os.path.isdir(img_dir):
            images.extend(sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                 if os.path.splitext(f)[1].lower() in IMG_EXT))
        # masks
        mask_dir = os.path.join(seg, "masks")
        if os.path.isdir(mask_dir):
            masks.extend(sorted(os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                                if os.path.splitext(f)[1].lower() in IMG_EXT))
        # bounding boxes
        bb = os.path.join(seg, "bounding-boxes.json")
        if os.path.isfile(bb):
            try:
                with open(bb, "r", encoding="utf-8") as fh:
                    bbox_meta.update(json.load(fh))
            except Exception as e:  # pragma: no cover
                print(f"[warn] could not parse {bb}: {e}", file=sys.stderr)
    return images, masks, bbox_meta


# ---------------------------------------------------------------------------
# Inventory + imbalance metrics
# ---------------------------------------------------------------------------
def build_inventory(records: List[ImageRecord]) -> pd.DataFrame:
    rows = [{"tract": r.tract, "category": r.category,
             "class_name": r.class_name, "split": r.split, "path": r.path}
            for r in records]
    return pd.DataFrame(rows)


def shannon_entropy(counts: Iterable[int]) -> float:
    c = np.asarray(list(counts), dtype=float)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gini_index(counts: Iterable[int]) -> float:
    """Gini coefficient of class frequencies (0 = perfectly balanced)."""
    c = np.sort(np.asarray(list(counts), dtype=float))
    n = len(c)
    if n == 0 or c.sum() == 0:
        return 0.0
    cum = np.cumsum(c)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def effective_num_samples(counts: Iterable[int], beta: float = 0.999) -> np.ndarray:
    """Cui et al. 2019, Class-Balanced Loss: (1 - beta^n) / (1 - beta)."""
    c = np.asarray(list(counts), dtype=float)
    return (1.0 - np.power(beta, c)) / (1.0 - beta)


def imbalance_report(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby("class_name").size().sort_values(ascending=False)
    total = counts.sum()
    rep = pd.DataFrame({
        "count": counts,
        "fraction": counts / total,
        "effective_n_beta0.999": effective_num_samples(counts.values),
    })
    rep.to_csv(out / "class_imbalance.csv")

    # Summary metrics file
    head_to_tail = float(counts.iloc[0] / max(counts.iloc[-1], 1))
    summary = {
        "n_classes": int(len(counts)),
        "n_images": int(total),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "head_to_tail_ratio": head_to_tail,
        "shannon_entropy_bits": shannon_entropy(counts.values),
        "max_entropy_bits": math.log2(len(counts)) if len(counts) > 1 else 0.0,
        "normalized_entropy": shannon_entropy(counts.values) / math.log2(len(counts))
                              if len(counts) > 1 else 0.0,
        "gini": gini_index(counts.values),
    }
    with open(out / "imbalance_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax, color="#3a7ca5")
    ax.set_ylabel("# images")
    ax.set_title(f"Class distribution (n={total}, "
                 f"head/tail={head_to_tail:.0f}x, "
                 f"Gini={summary['gini']:.2f})")
    ax.set_xlabel("")
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image statistics
# ---------------------------------------------------------------------------
def _specular_fraction(arr_rgb: np.ndarray, thresh: int = 240) -> float:
    """Fraction of pixels above ``thresh`` in *all* channels -- a coarse proxy
    for instrument-light glare on glossy mucosal tissue."""
    if arr_rgb.ndim != 3:
        return 0.0
    mask = (arr_rgb[..., 0] >= thresh) & (arr_rgb[..., 1] >= thresh) & (arr_rgb[..., 2] >= thresh)
    return float(mask.mean())


def per_image_stats(records: List[ImageRecord], sample_size: int = 64) -> pd.DataFrame:
    """Per-image stats. ``sample_size`` is the side length we downsample to
    before computing color stats -- gives ~constant-time per image."""
    rows: List[dict] = []
    for r in tqdm(records, desc="image stats", unit="img"):
        try:
            with Image.open(r.path) as im:
                w, h = im.size
                im_small = im.convert("RGB").resize((sample_size, sample_size),
                                                    Image.BILINEAR)
                arr = np.asarray(im_small)
            mean_rgb = arr.reshape(-1, 3).mean(axis=0)
            brightness = float(arr.mean())
            # rough saturation: max-min over channels
            sat = float((arr.max(axis=-1) - arr.min(axis=-1)).mean())
            rows.append({
                "path": r.path,
                "tract": r.tract, "category": r.category,
                "class_name": r.class_name, "split": r.split,
                "width": w, "height": h, "aspect": w / h if h else None,
                "mean_R": float(mean_rgb[0]),
                "mean_G": float(mean_rgb[1]),
                "mean_B": float(mean_rgb[2]),
                "brightness": brightness,
                "saturation_proxy": sat,
                "specular_frac": _specular_fraction(arr),
            })
        except Exception as e:  # pragma: no cover
            print(f"[warn] skip {r.path}: {e}", file=sys.stderr)
    return pd.DataFrame(rows)


def plot_image_stats(stats: pd.DataFrame, out: Path) -> None:
    if stats.empty:
        return
    # Resolution scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(stats["width"], stats["height"], s=6, alpha=0.4, color="#264653")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title(f"Image resolutions (n={len(stats)})")
    fig.tight_layout()
    fig.savefig(out / "resolutions.png", dpi=150)
    plt.close(fig)

    # Brightness / saturation / specular fraction distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes,
                              ["brightness", "saturation_proxy", "specular_frac"],
                              ["#e76f51", "#f4a261", "#2a9d8f"]):
        ax.hist(stats[col].dropna(), bins=40, color=color)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(out / "image_color_stats.png", dpi=150)
    plt.close(fig)

    # Per-class brightness / specular fraction (boxplots)
    for col in ["brightness", "specular_frac"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        order = (stats.groupby("class_name")[col].median().sort_values().index)
        data = [stats.loc[stats["class_name"] == c, col].values for c in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title(f"{col} per class (medians sorted)")
        plt.xticks(rotation=75, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"per_class_{col}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Duplicate / near-duplicate detection
# ---------------------------------------------------------------------------
def perceptual_dedup(records: List[ImageRecord], hash_size: int = 8,
                     hamming_thresh: int = 5) -> pd.DataFrame:
    """Compute pHash per image and report near-duplicate clusters."""
    if not _HAS_IMAGEHASH:
        print("[warn] imagehash not installed; skipping dedup", file=sys.stderr)
        return pd.DataFrame()
    hashes: Dict[str, "imagehash.ImageHash"] = {}
    for r in tqdm(records, desc="phash", unit="img"):
        try:
            with Image.open(r.path) as im:
                hashes[r.path] = imagehash.phash(im, hash_size=hash_size)
        except Exception as e:
            print(f"[warn] phash skip {r.path}: {e}", file=sys.stderr)
    # naive O(n^2) clustering -- fine for ~2k images
    paths = list(hashes.keys())
    groups: List[List[str]] = []
    assigned = [False] * len(paths)
    for i, p in enumerate(paths):
        if assigned[i]:
            continue
        cluster = [p]
        assigned[i] = True
        for j in range(i + 1, len(paths)):
            if assigned[j]:
                continue
            if hashes[p] - hashes[paths[j]] <= hamming_thresh:
                cluster.append(paths[j])
                assigned[j] = True
        if len(cluster) > 1:
            groups.append(cluster)
    rows: List[dict] = []
    for gid, g in enumerate(groups):
        for p in g:
            rows.append({"cluster_id": gid, "size": len(g), "path": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------
def video_metadata(records: List[VideoRecord]) -> pd.DataFrame:
    if not _HAS_CV2:
        print("[warn] opencv-python not installed; skipping video metadata",
              file=sys.stderr)
        return pd.DataFrame()
    rows: List[dict] = []
    for r in tqdm(records, desc="videos", unit="vid"):
        try:
            cap = cv2.VideoCapture(r.path)
            if not cap.isOpened():
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            rows.append({
                "path": r.path, "tract": r.tract, "category": r.category,
                "class_name": r.class_name, "split": r.split,
                "fps": fps, "n_frames": nframes,
                "duration_s": nframes / fps if fps > 0 else None,
                "width": w, "height": h,
            })
        except Exception as e:  # pragma: no cover
            print(f"[warn] video skip {r.path}: {e}", file=sys.stderr)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segmentation stats
# ---------------------------------------------------------------------------
def segmentation_stats(images: List[str], masks: List[str],
                       bbox_meta: dict) -> pd.DataFrame:
    rows: List[dict] = []
    mask_by_stem = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
    for img_path in tqdm(images, desc="seg", unit="img"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        info: dict = {"path": img_path, "stem": stem}
        # bbox info
        bb = bbox_meta.get(stem)
        if bb is not None:
            info["n_bbox"] = len(bb.get("bbox", []))
            info["bbox_w"] = bb.get("width")
            info["bbox_h"] = bb.get("height")
            info["bbox_labels"] = ",".join(sorted({b["label"] for b in bb.get("bbox", [])}))
        # mask-area ratio
        mp = mask_by_stem.get(stem)
        if mp is not None:
            try:
                with Image.open(mp) as m:
                    marr = np.asarray(m.convert("L"))
                    info["mask_area_frac"] = float((marr > 0).mean())
            except Exception:
                pass
        rows.append(info)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="EDA for hkv_subsample_p2")
    ap.add_argument("--data", required=True,
                    help="Path to toyDataset (parent of the 3 split folders)")
    ap.add_argument("--out", default="./results/eda",
                    help="Output directory for CSVs and plots")
    ap.add_argument("--quick", action="store_true",
                    help="Skip per-image color stats and dedup")
    ap.add_argument("--dedup-hamming", type=int, default=5,
                    help="Hamming-distance threshold for near-dup clusters (pHash)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    roots = find_split_roots(args.data)
    if not roots:
        raise SystemExit(f"No hkv_subsample_p2 split folders found under {args.data}")
    print(f"[info] discovered splits: {[s for s, _ in roots]}")

    img_records = discover_images(roots)
    vid_records = discover_videos(roots)
    seg_imgs, seg_masks, bbox_meta = discover_segmented(roots)
    print(f"[info] images: {len(img_records)}  videos: {len(vid_records)}  "
          f"seg-images: {len(seg_imgs)}  masks: {len(seg_masks)}  "
          f"bbox entries: {len(bbox_meta)}")

    # 1. Inventory
    inv = build_inventory(img_records)
    inv.to_csv(out / "inventory_images.csv", index=False)
    inv.groupby(["tract", "category", "class_name"]).size().to_csv(
        out / "counts_by_class.csv", header=["count"])

    # 2. Imbalance
    if not inv.empty:
        imbalance_report(inv, out)

    # 3. Image statistics
    if not args.quick and img_records:
        stats = per_image_stats(img_records)
        stats.to_csv(out / "image_stats.csv", index=False)
        plot_image_stats(stats, out)
        # Per-class summary
        if not stats.empty:
            stats.groupby("class_name").agg(
                n=("path", "count"),
                w_med=("width", "median"), h_med=("height", "median"),
                brightness_med=("brightness", "median"),
                spec_med=("specular_frac", "median"),
            ).to_csv(out / "image_stats_per_class.csv")

    # 4. Duplicate detection
    if not args.quick and img_records:
        dups = perceptual_dedup(img_records, hamming_thresh=args.dedup_hamming)
        dups.to_csv(out / "near_duplicates.csv", index=False)
        if not dups.empty:
            print(f"[info] near-duplicate clusters: {dups['cluster_id'].nunique()}"
                  f" (avg size {dups.groupby('cluster_id').size().mean():.2f})")

    # 5. Videos
    if vid_records:
        vmeta = video_metadata(vid_records)
        vmeta.to_csv(out / "video_metadata.csv", index=False)
        if not vmeta.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vmeta["duration_s"].dropna(), bins=30, color="#9d4edd")
            ax.set_xlabel("duration (s)")
            ax.set_title("Video duration distribution")
            fig.tight_layout()
            fig.savefig(out / "video_durations.png", dpi=150)
            plt.close(fig)

    # 6. Segmentation
    if seg_imgs:
        seg_df = segmentation_stats(seg_imgs, seg_masks, bbox_meta)
        seg_df.to_csv(out / "segmentation_stats.csv", index=False)
        if "mask_area_frac" in seg_df and seg_df["mask_area_frac"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(seg_df["mask_area_frac"].dropna(), bins=30, color="#06a77d")
            ax.set_xlabel("mask area fraction")
            ax.set_title("Polyp mask area / image area")
            fig.tight_layout()
            fig.savefig(out / "mask_area_frac.png", dpi=150)
            plt.close(fig)

    # 7. Markdown summary
    summary_md = out / "SUMMARY.md"
    with open(summary_md, "w", encoding="utf-8") as fh:
        fh.write("# EDA summary -- hkv_subsample_p2\n\n")
        fh.write(f"* images: **{len(img_records)}**\n")
        fh.write(f"* videos: **{len(vid_records)}**\n")
        fh.write(f"* segmentation images: **{len(seg_imgs)}**, masks: {len(seg_masks)}, "
                 f"bbox entries: {len(bbox_meta)}\n\n")
        if not inv.empty:
            fh.write("See `class_imbalance.csv`, `imbalance_summary.json`, "
                     "and the PNG plots in this folder.\n")
    print(f"[done] outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
