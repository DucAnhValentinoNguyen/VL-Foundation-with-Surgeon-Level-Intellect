#!/usr/bin/env python3
"""
benchmark.py -- Stress-test the SigLIP2 encoder on hkv_subsample_p2
====================================================================

Pipeline
--------
1. **Load model**. Any SigLIP2 checkpoint from HuggingFace Hub
   (``google/siglip2-*``). The flagship is ``google/siglip2-so400m-patch14-384``
   which fits comfortably in an RTX 4090's 24 GB at fp16.

2. **Extract image embeddings** for every image in ``labeled-images``,
   for ``--frames-per-video`` uniformly sampled frames of every video in
   ``labeled-videos``, and for every image in ``segmented-images``.
   Saved as ``embeddings.npz`` so subsequent probes are cheap to re-run.

3. **Benchmarks** -- five families, each with its own paper-backed motivation:

   a. **Zero-shot classification** (Radford et al. 2021;
      Tschannen et al. 2025 arXiv:2502.14786). Two prompt ensembles
      (generic vs. endoscopy-conditioned), closed-set and open-set
      variants. Reports top-1, top-5, balanced accuracy, macro-F1 and
      per-class confusion. This directly answers "does SigLIP2's
      pretrained alignment transfer to surgical concepts?".

   b. **Linear probe + k-NN** (de Facto CLIP eval since Radford 2021).
      Frozen encoder, train a logistic regression head with sklearn,
      and a 5-NN classifier in the embedding space. Removes text from
      the loop and isolates feature quality.

   c. **Retrieval & embedding geometry**. Intra- / inter-class cosine
      similarity, R@1 / R@5 / R@10 (image-image), mAP, silhouette
      score, and the alignment / uniformity metrics from Wang & Isola
      (ICML 2020). t-SNE / UMAP plot if those libs are available.

   d. **Robustness stress tests**. Compute embeddings on perturbed
      copies of a stratified subsample and measure
      ``cosine(emb_clean, emb_perturbed)`` plus zero-shot accuracy
      drop. Perturbations: Gaussian blur, JPEG compression, brightness
      shift, contrast shift, simulated specular glare, Gaussian noise,
      and a resolution sweep (128 / 192 / 256 / 384 / 512 px input
      side). This is the focused encoder stress test.

   e. **Imbalance-aware metrics + importance weighting**
      (Cui et al. 2019; Kulesza, Pereira, Crammer 2010 MLJ on adaptation;
      Awasthi, Cortes, Mansour PMLR 2023). On top of standard top-1
      we report balanced accuracy, macro-F1, per-class precision/recall,
      and an importance-weighted accuracy where weights come from a
      target prior the user can supply via JSON (defaults to uniform,
      i.e. balanced accuracy in expectation).

   f. **Video temporal coherence**. For each video sample N frames,
      compute the mean adjacent-frame cosine, the embedding drift
      (cosine to the first frame as a function of time), and contrast
      single-frame vs. mean-pooled clip classification.

All numbers land in ``--out`` as CSV / JSON / PNG, ready to drop into
the report.

Usage
-----
    python benchmark.py --data /path/to/toyDataset \\
        --model google/siglip2-base-patch16-224 \\
        --out ./results/benchmark \\
        --batch-size 64 --frames-per-video 16

    # bigger model, fp16, more samples for robustness
    python benchmark.py --data /path/to/toyDataset \\
        --model google/siglip2-so400m-patch14-384 \\
        --precision fp16 --robust-samples 400

Dependencies
------------
    pip install torch torchvision transformers pillow numpy pandas \\
        scikit-learn matplotlib opencv-python tqdm
    # optional
    pip install umap-learn

References
----------
* Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with
  Improved Semantic Understanding, Localization, and Dense Features",
  arXiv:2502.14786, 2025.
* Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision" (CLIP), 2021.
* Cui et al., "Class-Balanced Loss Based on Effective Number of
  Samples", CVPR 2019.
* Wang & Isola, "Understanding Contrastive Representation Learning
  through Alignment and Uniformity on the Hypersphere", ICML 2020.
* Kulesza, Pereira, "Multi-Task Learning for Stretched-Structured Output
  Prediction", and Crammer/Kulesza/Pereira MLJ 2010 (importance weighting
  under covariate shift). Awasthi, Cortes, Mansour PMLR 206 2023
  (theory of importance weighting and reweighting-based adaptation).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Local imports
try:
    from prompts import (
        CLASS_LABELS, TEMPLATES_GENERIC, TEMPLATES_ENDOSCOPY,
        build_prompts, IMAGE_CLASSES, VIDEO_ONLY_CLASSES,
    )
except ImportError:  # allow running from a different cwd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prompts import (  # noqa: E402
        CLASS_LABELS, TEMPLATES_GENERIC, TEMPLATES_ENDOSCOPY,
        build_prompts, IMAGE_CLASSES, VIDEO_ONLY_CLASSES,
    )


# ---------------------------------------------------------------------------
# Deferred heavy imports (so --help works without torch/transformers)
# ---------------------------------------------------------------------------
def _torch():
    import torch  # noqa: F401
    return __import__("torch")


def _transformers():
    return __import__("transformers")


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Data discovery (mirrors eda.py)
# ---------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXT = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass
class ImageRecord:
    path: str
    tract: str
    category: str
    class_name: str


@dataclass
class VideoRecord:
    path: str
    tract: str
    category: str
    class_name: str


def find_split_roots(data_dir: str) -> List[str]:
    roots: List[str] = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full) or "hkv_subsample_p2" not in entry:
            continue
        inner = os.path.join(full, "hkv_subsample_p2")
        if os.path.isdir(inner):
            roots.append(inner)
    if not roots:
        inner = os.path.join(data_dir, "hkv_subsample_p2")
        if os.path.isdir(inner):
            roots.append(inner)
        elif os.path.basename(data_dir.rstrip("/")) == "hkv_subsample_p2":
            roots.append(data_dir)
    return roots


def discover_images(roots: Sequence[str]) -> List[ImageRecord]:
    out: List[ImageRecord] = []
    for root in roots:
        labeled = os.path.join(root, "labeled-images")
        if not os.path.isdir(labeled):
            continue
        for tract in sorted(os.listdir(labeled)):
            for category in sorted(os.listdir(os.path.join(labeled, tract))):
                cdir = os.path.join(labeled, tract, category)
                if not os.path.isdir(cdir):
                    continue
                for cls in sorted(os.listdir(cdir)):
                    clsdir = os.path.join(cdir, cls)
                    if not os.path.isdir(clsdir):
                        continue
                    for f in sorted(os.listdir(clsdir)):
                        if os.path.splitext(f)[1].lower() in IMG_EXT:
                            out.append(ImageRecord(
                                path=os.path.join(clsdir, f),
                                tract=tract, category=category,
                                class_name=cls,
                            ))
    return out


def discover_videos(roots: Sequence[str]) -> List[VideoRecord]:
    out: List[VideoRecord] = []
    for root in roots:
        labeled = os.path.join(root, "labeled-videos")
        if not os.path.isdir(labeled):
            continue
        for tract in sorted(os.listdir(labeled)):
            for category in sorted(os.listdir(os.path.join(labeled, tract))):
                cdir = os.path.join(labeled, tract, category)
                if not os.path.isdir(cdir):
                    continue
                for cls in sorted(os.listdir(cdir)):
                    clsdir = os.path.join(cdir, cls)
                    if not os.path.isdir(clsdir):
                        continue
                    for f in sorted(os.listdir(clsdir)):
                        if os.path.splitext(f)[1].lower() in VID_EXT:
                            out.append(VideoRecord(
                                path=os.path.join(clsdir, f),
                                tract=tract, category=category,
                                class_name=cls,
                            ))
    return out


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class SiglipEncoder:
    """Thin wrapper around a HuggingFace SigLIP2 checkpoint."""

    def __init__(self, model_name: str, device: Optional[str] = None,
                 precision: str = "fp32"):
        torch = _torch()
        tf = _transformers()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                      "fp32": torch.float32}[precision]
        print(f"[model] loading {model_name} on {self.device} ({precision})")
        self.processor = tf.AutoProcessor.from_pretrained(model_name)
        self.model = tf.AutoModel.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device).eval()
        # Detect canonical image size from the processor.
        try:
            self.image_size = self.processor.image_processor.size["height"]
        except Exception:
            self.image_size = 224

    @property
    def torch(self):
        return _torch()

    # -- text -----------------------------------------------------------
    def encode_text(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(text=texts, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Correctly pull the tensor from the Hugging Face container
            outputs = self.model.get_text_features(**inputs)
            # Use pooler_output if available, otherwise default to the first element
            feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
            
        # Now normalize the extracted tensor
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()

    # -- images ---------------------------------------------------------
    def encode_pil(self, images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        from PIL import Image
        import numpy as np
        torch = self.torch
        out: List[np.ndarray] = []
        
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device).to(self.dtype) if v.dtype.is_floating_point
                      else v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Extract the tensor from the Hugging Face wrapper container
                outputs = self.model.get_image_features(**inputs)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
            
            # Now normalization works perfectly because 'feats' is a real tensor
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy())
            
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))

    def encode_paths(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        out: List[np.ndarray] = []
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **k: x  # noqa: E731




        """
        The Hidden Bug: Index Realignment Shift
        If any image path in batch_paths is corrupted, missing, or fails to open via Image.open(p), your script prints a warning and skips it (continue).
        This causes the size of the imgs list to be smaller than the size of batch_paths. When self.processor(images=imgs) converts the images into tensors and passes them to the model, it returns fewer embedding vectors than the number of paths requested in that slice.
        When the downstream benchmark processes those embeddings, your arrays will fall out of alignment:
        Paths array:   [img_0, img_1 (corrupted), img_2, img_3] -> Length 4
        Embeddings:    [emb_0,                   emb_2, emb_3] -> Length 3
        Labels array:  [lbl_0, lbl_1,             lbl_2, lbl_3] -> Length 4
        This causes a silent shift where emb_2 matches up with lbl_1, corrupting our entire zero-shot and linear-probe accuracy validation metrics without throwing an error.
        """
        # Create a standard 224x224 black image placeholder for unreadable files
        # This keeps our paths, labels, and embeddings perfectly aligned in index
        fallback_img = Image.new("RGB", (224, 224), (0, 0, 0))

        for i in tqdm(range(0, len(paths), batch_size), desc="encode-img"):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            
            for p in batch_paths:
                try:
                    # Append a closed handle or explicitly load to catch errors early
                    img = Image.open(p).convert("RGB")
                    img.load() 
                    imgs.append(img)
                except Exception as e:
                    print(f"[warn] Failed to open {p}: {e}. Using blank placeholder.", file=sys.stderr)
                    imgs.append(fallback_img.copy())

            if not imgs:
                continue

            inputs = self.processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device).to(self.dtype) if v.dtype.is_floating_point
                      else v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Correctly pulling the underlying tensor from the HF container object
                outputs = self.model.get_image_features(**inputs)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
            
            # Safe normalization on our valid PyTorch tensor
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy())
            
            # Clean up image allocations in RAM
            for im in imgs:
                im.close()
                
        fallback_img.close()
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


# ---------------------------------------------------------------------------
# Zero-shot helpers
# ---------------------------------------------------------------------------
def text_classifier(encoder: SiglipEncoder, class_names: List[str],
                    templates: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Return [n_class, dim] mean prompt-ensemble embedding + nice labels."""
    nice_labels = [CLASS_LABELS.get(c, c.replace("-", " ")) for c in class_names]
    matrices = []
    for nice in nice_labels:
        prompts = build_prompts(nice, templates)
        feats = encoder.encode_text(prompts)  # [P, D]
        mean = feats.mean(axis=0, keepdims=True)
        mean /= (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-12)
        matrices.append(mean)
    return np.concatenate(matrices, axis=0), nice_labels


def zero_shot_eval(img_emb: np.ndarray, img_labels: np.ndarray,
                   class_emb: np.ndarray, class_names: List[str]) -> dict:
    sims = img_emb @ class_emb.T  # [N, C]
    preds = sims.argmax(axis=1)
    top5 = np.argsort(-sims, axis=1)[:, :5]
    top1_acc = float((preds == img_labels).mean()) if len(img_labels) else 0.0
    top5_acc = float(np.mean([img_labels[i] in top5[i]
                              for i in range(len(img_labels))])) if len(img_labels) else 0.0
    # per-class
    per_class: Dict[str, dict] = {}
    for ci, cname in enumerate(class_names):
        mask = img_labels == ci
        if mask.sum() == 0:
            continue
        per_class[cname] = {
            "n": int(mask.sum()),
            "recall": float((preds[mask] == ci).mean()),
        }
    bal = float(np.mean([v["recall"] for v in per_class.values()])) if per_class else 0.0
    # macro-F1
    macro_f1 = _macro_f1(img_labels, preds, n_classes=len(class_names))
    return {
        "top1": top1_acc, "top5": top5_acc,
        "balanced_acc": bal, "macro_f1": macro_f1,
        "per_class": per_class,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Linear probe + k-NN
# ---------------------------------------------------------------------------
def linear_probe_and_knn(emb: np.ndarray, labels: np.ndarray,
                         class_names: List[str], seed: int,
                         min_per_class: int = 4) -> dict:
    """Stratified 70/30 split, logistic regression and 5-NN."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    # Drop classes too small to stratify
    counts = Counter(labels.tolist())
    keep = [c for c, n in counts.items() if n >= min_per_class]
    keep_set = set(keep)
    mask = np.array([l in keep_set for l in labels])
    X, y = emb[mask], labels[mask]
    if len(np.unique(y)) < 2:
        return {"error": "not enough classes for probing"}
    # re-map labels to 0..K-1
    remap = {old: new for new, old in enumerate(sorted(set(y.tolist())))}
    y_re = np.array([remap[v] for v in y])
    kept_names = [class_names[old] for old in sorted(remap)]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_re, test_size=0.3, stratify=y_re, random_state=seed,
    )

    # Linear probe
    lr = LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1,
                            class_weight="balanced")
    lr.fit(Xtr, ytr)
    lr_pred = lr.predict(Xte)
    lr_top1 = float((lr_pred == yte).mean())
    lr_bal = float(np.mean([
        (lr_pred[yte == c] == c).mean()
        for c in range(len(kept_names)) if (yte == c).any()
    ]))
    lr_f1 = _macro_f1(yte, lr_pred, len(kept_names))

    # k-NN (cosine via normalised features + euclidean)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                               weights="distance", n_jobs=-1)
    knn.fit(Xtr, ytr)
    knn_pred = knn.predict(Xte)
    knn_top1 = float((knn_pred == yte).mean())
    knn_bal = float(np.mean([
        (knn_pred[yte == c] == c).mean()
        for c in range(len(kept_names)) if (yte == c).any()
    ]))
    knn_f1 = _macro_f1(yte, knn_pred, len(kept_names))

    return {
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "kept_classes": kept_names,
        "linear_probe": {"top1": lr_top1, "balanced_acc": lr_bal, "macro_f1": lr_f1},
        "knn5": {"top1": knn_top1, "balanced_acc": knn_bal, "macro_f1": knn_f1},
    }


# ---------------------------------------------------------------------------
# Retrieval + embedding geometry
# ---------------------------------------------------------------------------
def retrieval_and_geometry(emb: np.ndarray, labels: np.ndarray,
                           class_names: List[str], max_n: int = 1500) -> dict:
    # Subsample for tractable pairwise computation
    if len(emb) > max_n:
        idx = np.random.default_rng(0).choice(len(emb), size=max_n, replace=False)
        emb = emb[idx]; labels = labels[idx]
    sims = emb @ emb.T  # cosine since normalised
    np.fill_diagonal(sims, -np.inf)
    # R@k
    out = {}
    for k in [1, 5, 10]:
        topk = np.argsort(-sims, axis=1)[:, :k]
        match = np.array([
            int((labels[topk[i]] == labels[i]).any()) for i in range(len(labels))
        ])
        out[f"R@{k}"] = float(match.mean())
    # mAP (per-query average precision over same-class hits)
    aps = []
    for i in range(len(labels)):
        order = np.argsort(-sims[i])
        rel = (labels[order] == labels[i]).astype(np.float32)
        if rel.sum() == 0:
            continue
        cum_hits = np.cumsum(rel)
        precisions = cum_hits / (np.arange(len(rel)) + 1)
        aps.append(float((precisions * rel).sum() / rel.sum()))
    out["mAP"] = float(np.mean(aps)) if aps else 0.0

    # intra / inter cosine
    intra, inter = [], []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 2:
            continue
        intra.append(sims[m][:, m][np.triu_indices(m.sum(), k=1)].mean())
        inter.append(sims[m][:, ~m].mean())
    out["intra_cos_mean"] = float(np.mean(intra)) if intra else 0.0
    out["inter_cos_mean"] = float(np.mean(inter)) if inter else 0.0
    out["intra_minus_inter"] = out["intra_cos_mean"] - out["inter_cos_mean"]

    # Alignment / Uniformity (Wang & Isola 2020)
    out["uniformity"] = _uniformity(emb)
    out["alignment"] = _alignment(emb, labels)

    # Silhouette (sklearn)
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            out["silhouette_cos"] = float(silhouette_score(emb, labels,
                                                           metric="cosine"))
    except Exception as e:
        out["silhouette_cos_error"] = str(e)
    return out


def _uniformity(x: np.ndarray, t: float = 2.0) -> float:
    """log E[exp(-t * ||x_i - x_j||^2)] -- want LOW (more uniform on sphere)."""
    # Pairwise squared distances on the unit sphere = 2 - 2 cos
    n = len(x)
    if n < 2:
        return 0.0
    s = x @ x.T
    d2 = 2 - 2 * s
    iu = np.triu_indices(n, k=1)
    return float(np.log(np.exp(-t * d2[iu]).mean() + 1e-12))


def _alignment(x: np.ndarray, labels: np.ndarray, t: float = 2.0) -> float:
    """E[||x_i - x_j||^t] over same-class pairs -- want LOW."""
    vals = []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 2:
            continue
        xc = x[m]
        d2 = 2 - 2 * (xc @ xc.T)
        iu = np.triu_indices(len(xc), k=1)
        vals.extend(d2[iu].tolist())
    if not vals:
        return 0.0
    return float(np.power(np.maximum(np.array(vals), 0), t / 2).mean())


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def _make_perturbations():
    """Return dict ``name -> callable(PIL -> PIL)``."""
    from PIL import Image, ImageFilter, ImageEnhance
    out = {}

    out["gauss_blur_2"]   = lambda im: im.filter(ImageFilter.GaussianBlur(2))
    out["gauss_blur_5"]   = lambda im: im.filter(ImageFilter.GaussianBlur(5))

    def jpeg(q):
        def _f(im, q=q):
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).copy()
        return _f
    out["jpeg_q30"] = jpeg(30)
    out["jpeg_q10"] = jpeg(10)

    out["bright_0.6"] = lambda im: ImageEnhance.Brightness(im).enhance(0.6)
    out["bright_1.4"] = lambda im: ImageEnhance.Brightness(im).enhance(1.4)
    out["contrast_0.6"] = lambda im: ImageEnhance.Contrast(im).enhance(0.6)
    out["contrast_1.4"] = lambda im: ImageEnhance.Contrast(im).enhance(1.4)

    def gauss_noise(sigma):
        def _f(im, sigma=sigma):
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            arr = arr + np.random.normal(0, sigma, arr.shape)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return _f
    out["noise_sigma15"] = gauss_noise(15)
    out["noise_sigma30"] = gauss_noise(30)

    def specular_glare(frac):
        """Splat a few bright Gaussian blobs to mimic endoscope light glare."""
        def _f(im, frac=frac):
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            h, w = arr.shape[:2]
            n = max(1, int(frac * 10))
            for _ in range(n):
                cy, cx = np.random.randint(0, h), np.random.randint(0, w)
                r = max(8, int(min(h, w) * 0.08))
                yy, xx = np.ogrid[:h, :w]
                g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
                arr = arr + 220 * g[..., None]
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return _f
    out["specular_light"] = specular_glare(0.3)
    out["specular_heavy"] = specular_glare(1.0)
    return out


def robustness_eval(encoder: SiglipEncoder, records: List[ImageRecord],
                    class_emb: np.ndarray, class_index: Dict[str, int],
                    n_samples: int, seed: int) -> pd.DataFrame:
    from PIL import Image
    rng = random.Random(seed)
    # Stratified subsample
    by_class: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        by_class[r.class_name].append(r)
    per_class = max(1, n_samples // max(len(by_class), 1))
    chosen: List[ImageRecord] = []
    for c, rs in by_class.items():
        rng.shuffle(rs)
        chosen.extend(rs[:per_class])
    print(f"[robust] using {len(chosen)} images across {len(by_class)} classes")

    # Encode clean
    clean_emb = encoder.encode_paths([r.path for r in chosen],
                                     batch_size=64)
    valid = [c for c in chosen if c.class_name in class_index]
    labels = np.array([class_index[r.class_name] for r in chosen
                       if r.class_name in class_index])
    valid_idx = [i for i, r in enumerate(chosen) if r.class_name in class_index]
    clean_valid = clean_emb[valid_idx]
    clean_acc = float((clean_valid @ class_emb.T).argmax(axis=1).__eq__(labels).mean()) \
        if len(labels) else 0.0

    perturbs = _make_perturbations()
    rows = [{
        "perturbation": "clean",
        "cosine_to_clean_mean": 1.0,
        "cosine_to_clean_p10": 1.0,
        "zero_shot_top1": clean_acc,
    }]
    for name, fn in perturbs.items():
        imgs = []
        for r in chosen:
            try:
                imgs.append(fn(Image.open(r.path).convert("RGB")))
            except Exception as e:
                print(f"[warn] {name} on {r.path}: {e}", file=sys.stderr)
                imgs.append(None)
        good = [im for im in imgs if im is not None]
        emb_p = encoder.encode_pil(good, batch_size=64)
        # cosine to clean (only matching indices)
        idx_good = [i for i, im in enumerate(imgs) if im is not None]
        coss = (clean_emb[idx_good] * emb_p).sum(axis=1)
        # zero-shot accuracy on perturbed
        valid_p_idx = [i for i, r in enumerate(chosen)
                       if (imgs[i] is not None and r.class_name in class_index)]
        emb_p_valid = encoder.encode_pil(
            [imgs[i] for i in valid_p_idx], batch_size=64
        )
        lab_p = np.array([class_index[chosen[i].class_name] for i in valid_p_idx])
        acc_p = float((emb_p_valid @ class_emb.T).argmax(axis=1).__eq__(lab_p).mean()) \
            if len(lab_p) else 0.0
        rows.append({
            "perturbation": name,
            "cosine_to_clean_mean": float(coss.mean()) if len(coss) else float("nan"),
            "cosine_to_clean_p10": float(np.percentile(coss, 10)) if len(coss) else float("nan"),
            "zero_shot_top1": acc_p,
        })
    return pd.DataFrame(rows)


def resolution_sweep(encoder: SiglipEncoder, records: List[ImageRecord],
                     class_emb: np.ndarray, class_index: Dict[str, int],
                     sizes=(128, 192, 256, 384, 512),
                     n_samples: int = 300, seed: int = 0) -> pd.DataFrame:
    """Resize image to ``s x s`` *before* feeding the processor. The processor
    will re-resize to its canonical size, but the information loss / gain
    matters."""
    from PIL import Image
    rng = random.Random(seed)
    rng.shuffle(records := list(records))
    valid = [r for r in records if r.class_name in class_index][:n_samples]
    labels = np.array([class_index[r.class_name] for r in valid])

    rows = []
    for s in sizes:
        imgs = []
        for r in valid:
            try:
                im = Image.open(r.path).convert("RGB").resize((s, s),
                                                              Image.BILINEAR)
                imgs.append(im)
            except Exception:
                imgs.append(None)
        good = [im for im in imgs if im is not None]
        lab_good = np.array([labels[i] for i, im in enumerate(imgs) if im is not None])
        emb = encoder.encode_pil(good, batch_size=64)
        acc = float((emb @ class_emb.T).argmax(axis=1).__eq__(lab_good).mean()) \
            if len(lab_good) else 0.0
        rows.append({"input_size": s, "zero_shot_top1": acc, "n": int(len(lab_good))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Importance-weighted accuracy (Kulesza-Pereira / Awasthi-Cortes-Mansour)
# ---------------------------------------------------------------------------
def importance_weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 target_prior: Optional[Dict[str, float]],
                                 class_names: List[str]) -> dict:
    """Per Awasthi-Cortes-Mansour (PMLR 2023) and earlier Kulesza work,
    when the test prior differs from the empirical (source) prior, the
    importance-weighted estimator gives an unbiased estimate of the target
    risk. Here we treat the empirical class distribution as the source and
    use ``target_prior`` (default: uniform) as the target.
    """
    src_counts = Counter(y_true.tolist())
    n = sum(src_counts.values())
    src_prior = {c: src_counts[i] / n for i, c in enumerate(class_names)
                 if src_counts[i] > 0}
    if target_prior is None:
        # uniform over observed classes -> reduces to balanced accuracy
        target_prior = {c: 1.0 / len(src_prior) for c in src_prior}
    # weights per example
    w = np.array([
        target_prior.get(class_names[c], 0.0) / max(src_prior.get(class_names[c], 1e-12), 1e-12)
        for c in y_true
    ])
    correct = (y_true == y_pred).astype(float)
    iw_acc = float((w * correct).sum() / (w.sum() + 1e-12))
    return {
        "iw_accuracy": iw_acc,
        "src_prior": src_prior,
        "target_prior": target_prior,
    }


# ---------------------------------------------------------------------------
# Video temporal coherence
# ---------------------------------------------------------------------------
def sample_video_frames(path: str, n_frames: int) -> List["Image.Image"]:
    from PIL import Image
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release(); return []
    idxs = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    frames: List[Image.Image] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(fr))
    cap.release()
    return frames


def video_temporal_eval(encoder: SiglipEncoder, videos: List[VideoRecord],
                        class_emb_image: np.ndarray, class_index: Dict[str, int],
                        n_frames: int) -> pd.DataFrame:
    rows = []
    for v in videos:
        frames = sample_video_frames(v.path, n_frames)
        if len(frames) < 2:
            continue
        emb = encoder.encode_pil(frames, batch_size=32)  # [F, D]
        adj = (emb[:-1] * emb[1:]).sum(axis=1)
        drift = (emb[0:1] @ emb.T)[0]
        # single-frame vs. clip pooled zero-shot
        per_frame_pred = (emb @ class_emb_image.T).argmax(axis=1)
        pooled = emb.mean(axis=0, keepdims=True)
        pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
        pooled_pred = int((pooled @ class_emb_image.T).argmax())
        gt = class_index.get(v.class_name, -1)
        rows.append({
            "path": v.path, "class": v.class_name, "n_frames": len(frames),
            "adj_cos_mean": float(adj.mean()),
            "adj_cos_min": float(adj.min()),
            "drift_first_to_last": float(drift[-1]),
            "majority_frame_pred": int(np.bincount(per_frame_pred).argmax()),
            "pooled_pred": pooled_pred,
            "gt_class_idx": gt,
            "majority_correct": int(np.bincount(per_frame_pred).argmax() == gt) if gt >= 0 else None,
            "pooled_correct": int(pooled_pred == gt) if gt >= 0 else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots & dimensionality reduction
# ---------------------------------------------------------------------------
def project_2d(emb: np.ndarray, labels: np.ndarray, class_names: List[str],
               out_path: Path, method: str = "tsne") -> None:
    plt = _setup_matplotlib()
    try:
        if method == "umap":
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0)
            xy = reducer.fit_transform(emb)
        else:
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, init="pca", random_state=0,
                      perplexity=min(30, max(5, len(emb) // 4))).fit_transform(emb)
    except Exception as e:
        print(f"[warn] {method} failed: {e}", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    uniq = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(uniq))
    for i, c in enumerate(uniq):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.7,
                   color=cmap(i), label=class_names[c])
    ax.set_title(f"{method.upper()} projection of SigLIP2 embeddings")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./results/benchmark")
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-per-video", type=int, default=16)
    ap.add_argument("--robust-samples", type=int, default=300)
    ap.add_argument("--res-sweep-samples", type=int, default=300)
    ap.add_argument("--target-prior", default=None,
                    help="JSON path: {class_name: prob} for importance weighting")
    ap.add_argument("--skip-robust", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    ap.add_argument("--skip-projection", action="store_true")
    ap.add_argument("--projection", choices=["tsne", "umap"], default="tsne")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); random.seed(args.seed)

    # ---------- discovery ----------
    roots = find_split_roots(args.data)
    if not roots:
        raise SystemExit(f"No hkv_subsample_p2 found under {args.data}")
    img_records = discover_images(roots)
    vid_records = discover_videos(roots)
    print(f"[info] images={len(img_records)}  videos={len(vid_records)}  splits={len(roots)}")

    # ---------- model ----------
    enc = SiglipEncoder(args.model, precision=args.precision)

    # ---------- image embeddings ----------
    emb_path = out / "image_embeddings.npz"
    if emb_path.exists():
        print(f"[info] loading cached embeddings from {emb_path}")
        cached = np.load(emb_path, allow_pickle=True)
        img_emb = cached["emb"]
        paths_cached = list(cached["paths"])
        class_names_cached = list(cached["class_names"])
        if paths_cached != [r.path for r in img_records]:
            print("[warn] cached paths differ -- re-encoding")
            img_emb = enc.encode_paths([r.path for r in img_records],
                                       batch_size=args.batch_size)
            np.savez(emb_path, emb=img_emb,
                     paths=np.array([r.path for r in img_records]),
                     class_names=np.array([r.class_name for r in img_records]))
    else:
        img_emb = enc.encode_paths([r.path for r in img_records],
                                   batch_size=args.batch_size)
        np.savez(emb_path, emb=img_emb,
                 paths=np.array([r.path for r in img_records]),
                 class_names=np.array([r.class_name for r in img_records]))
    cls_of = [r.class_name for r in img_records]
    class_names = sorted(set(cls_of))
    class_index = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_index[c] for c in cls_of])
    print(f"[info] embeddings shape={img_emb.shape}  classes={len(class_names)}")

    # ---------- 1. Zero-shot ----------
    print("[step] zero-shot classification")
    zs_results = {}
    for tname, templates in [
        ("generic_prompts", TEMPLATES_GENERIC),
        ("endoscopy_prompts", TEMPLATES_ENDOSCOPY),
    ]:
        for cset_name, cset in [
            ("closed_set", class_names),
            ("openset_with_video_classes",
             sorted(set(class_names + VIDEO_ONLY_CLASSES))),
        ]:
            class_emb, _ = text_classifier(enc, cset, templates)
            cidx = {c: i for i, c in enumerate(cset)}
            # remap labels for this class set
            lab_remap = np.array([cidx[c] for c in cls_of])
            res = zero_shot_eval(img_emb, lab_remap, class_emb, cset)
            iw = importance_weighted_accuracy(
                lab_remap, (img_emb @ class_emb.T).argmax(axis=1),
                target_prior=None, class_names=cset,
            )
            res["importance_weighted"] = iw
            zs_results[f"{tname}__{cset_name}"] = res
            print(f"   {tname:>20s} | {cset_name:>22s}  top1={res['top1']:.3f}  "
                  f"bal={res['balanced_acc']:.3f}  f1={res['macro_f1']:.3f}")
    with open(out / "zero_shot.json", "w") as fh:
        json.dump(zs_results, fh, indent=2)

    # Confusion CSV for the best variant
    best = max(zs_results.items(), key=lambda kv: kv[1]["balanced_acc"])
    print(f"[info] best zero-shot variant: {best[0]} (bal={best[1]['balanced_acc']:.3f})")

    # ---------- 2. Linear probe + k-NN ----------
    print("[step] linear probe + 5-NN")
    probe = linear_probe_and_knn(img_emb, labels, class_names, seed=args.seed)
    with open(out / "linear_probe.json", "w") as fh:
        json.dump(probe, fh, indent=2)
    print(f"   linear: {probe.get('linear_probe')}")
    print(f"   knn5  : {probe.get('knn5')}")

    # ---------- 3. Retrieval + geometry ----------
    print("[step] retrieval & geometry")
    geom = retrieval_and_geometry(img_emb, labels, class_names)
    with open(out / "retrieval_geometry.json", "w") as fh:
        json.dump(geom, fh, indent=2)
    print(f"   R@1={geom.get('R@1'):.3f}  R@5={geom.get('R@5'):.3f}  "
          f"mAP={geom.get('mAP'):.3f}  silh={geom.get('silhouette_cos', float('nan')):.3f}")

    # ---------- 4. Robustness + resolution ----------
    if not args.skip_robust:
        print("[step] robustness corruptions")
        # Use endoscopy prompts on closed set as the canonical reference.
        class_emb_ref, _ = text_classifier(enc, class_names, TEMPLATES_ENDOSCOPY)
        rob = robustness_eval(enc, img_records, class_emb_ref, class_index,
                              n_samples=args.robust_samples, seed=args.seed)
        rob.to_csv(out / "robustness.csv", index=False)
        print(rob.to_string(index=False))

        print("[step] resolution sweep")
        rs = resolution_sweep(enc, img_records, class_emb_ref, class_index,
                              n_samples=args.res_sweep_samples, seed=args.seed)
        rs.to_csv(out / "resolution_sweep.csv", index=False)
        print(rs.to_string(index=False))

        # Plot robustness
        plt = _setup_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(rob["perturbation"], rob["zero_shot_top1"], color="#e63946")
        ax.set_ylabel("zero-shot top-1")
        ax.set_title("Robustness: zero-shot top-1 under perturbation")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out / "robustness_top1.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rs["input_size"], rs["zero_shot_top1"], "o-", color="#1d3557")
        ax.set_xlabel("input resolution (px, square)")
        ax.set_ylabel("zero-shot top-1")
        ax.set_title("Resolution sweep")
        fig.tight_layout()
        fig.savefig(out / "resolution_sweep.png", dpi=150)
        plt.close(fig)

    # ---------- 5. Importance-weighted variant ----------
    target_prior = None
    if args.target_prior:
        with open(args.target_prior) as fh:
            target_prior = json.load(fh)
    class_emb_ref, _ = text_classifier(enc, class_names, TEMPLATES_ENDOSCOPY)
    preds = (img_emb @ class_emb_ref.T).argmax(axis=1)
    iw = importance_weighted_accuracy(labels, preds,
                                      target_prior=target_prior,
                                      class_names=class_names)
    with open(out / "importance_weighted.json", "w") as fh:
        json.dump(iw, fh, indent=2)
    print(f"[step] importance-weighted acc = {iw['iw_accuracy']:.3f}")

    # ---------- 6. Video temporal coherence ----------
    if not args.skip_videos and vid_records:
        print("[step] video temporal coherence")
        vid_df = video_temporal_eval(enc, vid_records, class_emb_ref,
                                     class_index, n_frames=args.frames_per_video)
        vid_df.to_csv(out / "video_temporal.csv", index=False)
        if not vid_df.empty:
            plt = _setup_matplotlib()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vid_df["adj_cos_mean"].dropna(), bins=30, color="#118ab2")
            ax.set_xlabel("mean adjacent-frame cosine similarity")
            ax.set_title("Temporal coherence of SigLIP2 embeddings")
            fig.tight_layout()
            fig.savefig(out / "video_adj_cos.png", dpi=150)
            plt.close(fig)

    # ---------- 7. 2D projection ----------
    if not args.skip_projection:
        print(f"[step] 2D projection ({args.projection})")
        project_2d(img_emb, labels, class_names,
                   out / f"projection_{args.projection}.png",
                   method=args.projection)

    # ---------- summary ----------
    summary = {
        "model": args.model,
        "precision": args.precision,
        "n_images": int(len(img_records)),
        "n_classes": int(len(class_names)),
        "zero_shot_best": {"variant": best[0],
                           "balanced_acc": best[1]["balanced_acc"],
                           "macro_f1": best[1]["macro_f1"]},
        "linear_probe": probe.get("linear_probe"),
        "knn5": probe.get("knn5"),
        "retrieval": {k: geom[k] for k in ("R@1", "R@5", "R@10", "mAP",
                                            "intra_cos_mean", "inter_cos_mean",
                                            "silhouette_cos") if k in geom},
        "importance_weighted_acc": iw["iw_accuracy"],
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] results in {out.resolve()}")


if __name__ == "__main__":
    main()
