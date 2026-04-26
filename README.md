# Vision Language Foundation with Surgeon Level Intellect: Technical Sprint Report
### Consulting Project | SS 2026 | LMU Munich

**Project Members:** 
* Lian Li 12871755
* Sahibnoor Singh 12690030
* Duc-Anh Nguyen 12433139


---


## 📌 Project Overview
The objective is to develop a functional prototype of a surgical vision-language foundation model capable of expert-level scene understanding and clinically grounded dialogue. This weekend, we built an end-to-end pipeline—from raw MP4 ingestion to Parameter-Efficient Fine-Tuning (PEFT)—demonstrating how a modern multimodal architecture (Qwen2.5-VL-3B) can be adapted to overcome the clinical hallucination gaps present in base foundation models. The hardware we used is an RTX 4090s with 24 GB of VRAM and 82.6 TFLOPS of FP32 compute. 

---

## 1. Model Architecture
Our project relies on the pre-trained **Qwen2.5-VL-3B** model as the foundation.  


## 2. Methodology, Results and Discussion: read our report.

---

## 🛠 Repository Structure
Following the mandatory course template:

```text
├── data/                # Dataset lists and loaders
├── src/                # Training scripts
├── results/            # Demo results
├── requirements.txt    # Environment dependencies
└── README.md

```
---

⚙️ Setup and Execution
For Task 3: Simultaneous assessment of instrument tracking precision, anatomical context, and clinical safety grounding.
Environment:

```bash
# install uv (if needed)
# for Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# install dependencies
uv pip install -r requirements.txt
```

Load Data
```bash
cd ./data/surgvu
# Download the validation set (includes clips already sampled at 1 FPS)
wget https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip
unzip cat1_test_set_public.zip
# Download the updated labels
wget https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip
unzip surgvu24_labels_updated_v2.zip
# Preprocess data
uv run dataset.py

Zero-shot
```bash
cd ../src & uv run baseline_eval.py
```

Fine-tuning:
```bash
uv run train_qlora.py
```

Evaluation:
```bash
uv run adapter_eval.py
uv run evaluate_metrics.py
# evaluate how the model's performance evolves over the duration of a surgical procedure
uv run temporal_analysis.py
```



