# Vision Language Foundation with Surgeon Level Intellect: Technical Sprint Report
### Consulting Project | SS 2026 | LMU Munich

**Project Members:** 
* Lian Li 
* Sahibnoor Singh 
* Duc-Anh Nguyen 


---


## 📌 Project Overview
The objective is to develop a functional prototype of a surgical vision-language foundation model capable of expert-level scene understanding and clinically grounded dialogue. This weekend, we built an end-to-end pipeline—from raw MP4 ingestion to Parameter-Efficient Fine-Tuning (PEFT)—demonstrating how a modern multimodal architecture (Qwen2.5-VL-3B) can be adapted to overcome the clinical hallucination gaps present in base foundation models. The hardware we used is an RTX 4090s with 24 GB of VRAM and 82.6 TFLOPS of FP32 compute. 

---

## 1. Model Architecture
Our project relies on the pre-trained **Qwen2.5-VL-3B** model as the foundation.  

---
## 2. Methodology
### Stage 1: Zero-Shot Learning


---

### Stage 2: Fine-Tuning Strategies


**Interpretation:** 


---

## 3. Discussion of Results and Limitations

### Results Analysis


### Limitations & Future Work

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
Environment:

```bash
pip install -r requirements.txt
```

Run Zero-Shot Baseline:
```bash
```


Fine-tuning:
```bash
```


