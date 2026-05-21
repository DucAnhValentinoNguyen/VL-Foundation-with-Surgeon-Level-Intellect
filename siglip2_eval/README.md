# SigLIP2 Evaluation Framework — `hkv_subsample_p2`

Phase 1 of the Zeiss consulting project: stress-test the **SigLIP2** image
encoder on real surgical / endoscopic data
(HyperKvasir subsample, three split archives).

The framework has two scripts and one helper module:

| File           | Purpose                                                             |
| -------------- | ------------------------------------------------------------------- |
| `eda.py`       | Exploratory data analysis — class imbalance, image stats, video metadata, segmentation, duplicate detection. |
| `benchmark.py` | Five-family benchmark — zero-shot, linear probe + k-NN, retrieval & geometry, robustness/resolution sweep, importance-weighted metrics, video temporal coherence. |
| `prompts.py`   | Canonical natural-language labels and prompt templates (generic + endoscopy-conditioned). |

## 1. Install

```bash
# Core
pip install torch torchvision transformers pillow numpy pandas \
            scikit-learn matplotlib opencv-python tqdm

# Optional (recommended)
pip install imagehash umap-learn
```

PyTorch must be the CUDA build matching your driver (CUDA 12.x for an RTX
4090 on a recent driver). HuggingFace download requires internet on the
first run; cache lives in `~/.cache/huggingface`.

## 2. Dataset layout

Point `--data` at the *parent* directory containing the three archive
folders (i.e. the `toyDataset/` you already have). Both scripts merge them
internally:

```
toyDataset/
├── hkv_subsample_p2-…-3-001/hkv_subsample_p2/{labeled-images, labeled-videos, segmented-images}
├── hkv_subsample_p2-…-3-002/hkv_subsample_p2/…
└── hkv_subsample_p2-…-3-003/hkv_subsample_p2/…
```

## 3. Run the EDA first

```bash
python eda.py --data /path/to/toyDataset --out ./results/eda
# fast pass, skip color stats + dedup:
python eda.py --data /path/to/toyDataset --out ./results/eda --quick
```

Outputs in `results/eda/`:

- `inventory_images.csv`, `counts_by_class.csv` — full inventory
- `class_imbalance.csv`, `imbalance_summary.json`, `class_distribution.png`
  — Shannon entropy, Gini, head/tail ratio, Cui-2019 effective n
- `image_stats.csv`, `image_stats_per_class.csv`,
  `resolutions.png`, `image_color_stats.png`,
  `per_class_brightness.png`, `per_class_specular_frac.png`
- `near_duplicates.csv` (pHash clustering, optional)
- `video_metadata.csv`, `video_durations.png`
- `segmentation_stats.csv`, `mask_area_frac.png`
- `SUMMARY.md`

## 4. Run the benchmark

```bash
# fastest baseline
python benchmark.py --data /path/to/toyDataset \
    --model google/siglip2-base-patch16-224 \
    --out ./results/bench_base

# flagship on the 4090 (≈22 GB VRAM in fp16, batched well)
python benchmark.py --data /path/to/toyDataset \
    --model google/siglip2-so400m-patch14-384 \
    --precision fp16 \
    --batch-size 32 \
    --frames-per-video 16 \
    --robust-samples 400 --res-sweep-samples 400 \
    --projection umap \
    --out ./results/bench_so400m
```

Optional flags:

- `--skip-robust`, `--skip-videos`, `--skip-projection` — for quick smoke runs
- `--target-prior PATH.json` — provide a `{class_name: probability}` map
  used as the *target* distribution for the importance-weighted accuracy
  (defaults to uniform → balanced accuracy in expectation, per
  Awasthi-Cortes-Mansour 2023 / Kulesza-Pereira reweighting framework).

Outputs in `results/bench_*/`:

- `image_embeddings.npz` — cached, re-used on subsequent runs
- `zero_shot.json` — 4 variants: `{generic|endoscopy} × {closed|openset}`
  with top-1, top-5, balanced acc, macro-F1, per-class recall
- `linear_probe.json` — frozen-feature logistic regression + 5-NN
- `retrieval_geometry.json` — R@1/5/10, mAP, intra/inter cosine,
  Wang-Isola alignment & uniformity, silhouette
- `robustness.csv`, `robustness_top1.png` — corruption suite
- `resolution_sweep.csv`, `resolution_sweep.png` — input-size sweep
- `importance_weighted.json`
- `video_temporal.csv`, `video_adj_cos.png` — adjacent-frame cosine,
  single-frame vs. clip-pooled classification
- `projection_tsne.png` or `projection_umap.png`
- `summary.json` — one-page roll-up

## 5. How the metrics map to the cited papers

| Metric / experiment                          | Paper                                                 |
| -------------------------------------------- | ----------------------------------------------------- |
| Prompt ensembling, zero-shot transfer        | Radford et al. 2021 (CLIP); Tschannen et al. 2025 (SigLIP2, arXiv:2502.14786) |
| Effective number of samples (β = 0.999)      | Cui et al. CVPR 2019                                  |
| Alignment / Uniformity on the hypersphere    | Wang & Isola, ICML 2020                               |
| Importance-weighted risk under prior shift   | Kulesza, Pereira et al. (MLJ 2010 adaptation paper); Awasthi, Cortes, Mansour, PMLR 206 (2023) |
| Macro-F1, balanced accuracy, per-class recall under imbalance | standard imbalanced-classification literature |
| Linear probing as a feature-quality probe    | Radford et al. 2021; Chen et al. 2020 (SimCLR)        |

## 6. Practical notes for the RTX 4090

- `--precision fp16` works well for all SigLIP2 sizes; `bf16` is slightly
  more numerically stable if you see NaNs.
- For `so400m-patch14-384` use `--batch-size 32`; for `base-patch16-224`
  push to 128.
- The embedding extraction is the slow step; subsequent re-runs reuse the
  cached `image_embeddings.npz`, so tweaking probes / robustness is fast.
- Videos read with OpenCV; install with `pip install opencv-python` (not
  `opencv-python-headless` if you want to be safe across hosts).

## 7. Known dataset caveats baked into the scripts

- **Classes with < 4 samples** are dropped from the linear probe (cannot
  stratify); they remain in the zero-shot eval, where per-class recall
  is reported.
- **Open-set zero-shot** adds the video-only classes (`cancer`, `ulcer`,
  `parasites`, …) to the label vocabulary even though no labeled *image*
  belongs to them. This tests whether the model spuriously attracts
  images to those textual concepts.
- The three split folders contain disjoint files (no overlap between
  archives) and we merge them by path before any train/test split.
