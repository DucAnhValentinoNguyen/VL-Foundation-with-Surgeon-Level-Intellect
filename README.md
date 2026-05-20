# Vision Language Foundation with Surgeon Level Intellect: Technical Report
### Consulting Project | SS 2026 | LMU Munich

**Project Members:** 
* Lian Li 12871755
* Sahibnoor Singh 12690030
* Duc-Anh Nguyen 12433139


---


## 📌 Project Overview
The objective is to develop a functional prototype of a surgical vision-language foundation model capable of expert-level scene understanding and clinically grounded dialogue.

---

## 1. Model Architecture
Our project relies on the pre-trained **SigLIP2** model as the foundation.  


## 2. Methodology, Results and Discussion: 
---

## 🛠 Repository Structure
Dataset for phase 1: testing SigLIP2 Encoder
```text

├──hkv_subsample_p2/
|    ├── labeled-images/ 
|    │   ├── lower-gi-tract/
|    |   │   ├── anatomical-landmarks/
|    |   │   │   ├── cecum/
|    |   │   │   ├── ileum/ 2 jpg files
|    |   │   │   └── retroflex-rectum/
|    |   │   ├── pathological-findings/
|    |   │   │   ├── hemorrhoids/ 1 jpg
|    |   │   │   ├── polyps/
|    |   │   │   ├── ulcerative-colitis-grade-0-1/
|    |   │   │   ├── ulcerative-colitis-grade-1/
|    |   │   │   ├── ulcerative-colitis-grade-1-2/
|    |   │   │   ├── ulcerative-colitis-grade-2/
|    |   │   │   ├── ulcerative-colitis-grade-2-3/
|    |   │   │   └── ulcerative-colitis-grade-3/
|    |   │   ├── quality-of-mucosal-views/
|    |   │   │   ├── bbps-0-1/
|    |   │   │   ├── bbps-2-3/
|    |   │   │   ├── impacted-stool/
|    |   │   │── therapeutic-interventions/
|    |   │   │   ├── dyed-lifted-polyps/
|    |   │   │   └── dyed-resection-margins/
|    |   │   └────
|    |   └────────
|    │   ├── upper-gi-tract/
|    |   │   ├── anatomical-landmarks/
|    |   │   │   ├── pylorus/
|    |   │   │   ├── retroflex-stomach/ 
|    |   │   │   └── z-line/
|    |   │   ├── pathological-findings/
|    |   │   │   ├── barrets/ 9 jpg
|    |   │   │   ├── barretts-short-segment/
|    |   │   │   ├── esophagitis-a/
|    |   │   │   └── esophagitis-b-d/
|    |   │   └────
|    |   └────────
|    └────────────
|    ├── labeled-videos/ 
|    │   ├── lower-gi-tract/
|    |   │   ├── anatomical-landmarks/
|    |   │   │   |
|    |   │   │   |
|    |   │   │   └── cecum/ 1 vid
|    |   │   ├── pathological-findings/
|    |   │   │   |
|    |   │   │   ├── polyps/
|    |   │   │   ├── colitis/ 3 vids
|    |   │   │   ├── anastomotic-leakage/ 1 vid
|    |   │   │   └── parasites/ 1 vid
|    |   │   ├── quality-of-mucosal-views/
|    |   │   │   ├── BBPS-0-1/ 2 vids
|    |   │   │   ├── BBPS-2-3/ 4 vids
|    |   │   │   └── 
|    |   │   │── therapeutic-interventions/
|    |   │   │   ├── dyed-lifted-polyps/ 7 vids
|    |   │   │   ├── dyed-resection-margins/ 2 vids
|    |   │   │   ├── self-expanding-stents/ 4 vids
|    |   │   │   └── snare-resection/ 2 vids
|    |   │   └────
|    |   └────────
|    │   ├── upper-gi-tract/
|    |   │   ├── anatomical-landmarks/
|    |   │   │   ├── 
|    |   │   │   ├── 
|    |   │   │   └── z-line/ 1 vid
|    |   │   ├── pathological-findings/
|    |   │   │   ├── 
|    |   │   │   ├──
|    |   │   │   ├── esophagitis/ 1 vid
|    |   │   │   ├── cancer/ 1 vid
|    |   │   │   ├── gastric-antral-vascular-ectasia/ 1 vid
|    |   │   │   ├── barretts-short-segment/ 1 vid
|    |   │   │   └── ulcer/ 2 vuds
|    |   │   └────
|    |   └────────
|    └────────────
|    ├── segmented-images/ 
|    │   ├── images/ # jpg files
|    │   ├── masks/ # jpg files
|    │   └── bounding-boxes.jsonl
|    |   │   └────
|    |   └────────
|    └────────────
└─────────────────


```
---

## ⚙️ Setup and Execution



