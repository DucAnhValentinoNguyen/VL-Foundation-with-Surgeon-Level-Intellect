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


## 2. Methodology, Results and Discussion: 
Please read our report in **Project Approach Proposal.pdf** 

---

## 🛠 Repository Structure
Following the mandatory course template:

```text
├──task1/
|    ├── data/ 
|    │   ├── testing/
|    │   ├── training/
├──task2/
|    ├── config.py
|    ├── requirements.txt
|    ├── notebooks/
|    ├── scripts/
|    │   ├── data/
|    │   ├── modeling/
|    │   ├── evaluation/
|    │   └── prompts/
|    ├── run_prepare_annotations.py
|    ├── run_teacher_label.py
|    ├── run_prepare_lora_data.py
|    ├── run_zero_shot.py
|    └── run_evaluation.py
├──task3/
|    ├── data/ 
|    │   ├── dataset.py
|    ├── src/ 
|    │   ├── baseline_eval.py
|    │   ├── evaluate_metrics.py
|    │   ├── finetuned_results.jsonl
|    │   ├── run_adapter_eval.py
|    │   ├── temporal_analysis.py
|    │   ├── temporal_segmentation.png
|    │   ├── train_qlora.py
|    │   ├── zero_shot_results.jsonl
|    ├── requirements.txt    # Environment dependencies
└── README.md

```
---

## ⚙️ Setup and Execution
### **For Task 1**: the implementation of the phase recognition pipeline, including zero-shot inference, fine-tuning, and evaluation.

Installation:

1. Clone the repository:
```bash
git clone https://github.com/DucAnhValentinoNguyen/VL-Foundation-with-Surgeon-Level-Intellect.git
```
2. Install dependencies:
```bash
pip install -r  task1/requirements.txt
```
3. Data setup

The data is distributed through a shared Google Drive folder. Please download or synchronize the data locally and place it under  the following structure before running the scripts:

```text
├──task1/
|    ├── data/ 
|    │   ├── testing/
|    │   ├── training/
```

Shared folder link:
https://drive.google.com/drive/folders/1zlW2PCKx1OrfQBOxMUCzln36k_G79Thc?usp=share_link


Execution:

1. Ground-truth generation
```bash
python task1/src/build_gt_jsonl.py
```
2. Zero-shot inference
```bash
python task1/src/run_zeroshot.py
```
3. Frozen-backbone fine-tuning
```bash
python task1/src/train_phase_head.py
```
4. LoRA-based fine-tuning
```bash
python task1/src/train_phase_lora.py
```
5. Fine-tuned inference
```bash
python task1/src/run_phase_finetuned_inference.py
```
6. Evaluation
```bash
python task1/src/eval_phase_predictions.py
```



### **For Task 2**: Expert-Level Surgical Communication


## Pipeline

```text
CholecSeg8k frame + annotation
        ↓
Extract visible classes
        ↓
Generate annotation-grounded teacher answer
        ↓
Prepare Qwen LoRA data
        ↓
Run Qwen zero-shot
        ↓
Evaluate
        ↓
Fine-tune with LoRA
        ↓
Compare zero-shot vs LoRA
````

---

## Dataset

Uses the sample version of **CholecSeg8k**.

Expected structure:

```text
task2/
├── ds/
│   ├── img/
│   ├── ann/
│   └── meta.json
```

Dataset split:

```text
Train: 674 frames
Validation: 146 frames
Test: 143 frames
```

---

## Main files

```text
task2/
├── config.py
├── requirements.txt
├── notebooks/
├── scripts/
│   ├── data/
│   ├── modeling/
│   ├── evaluation/
│   └── prompts/
├── run_prepare_annotations.py
├── run_teacher_label.py
├── run_prepare_lora_data.py
├── run_zero_shot.py
└── run_evaluation.py
```

---

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare data:

```bash
python run_prepare_annotations.py
python run_teacher_label.py
python run_prepare_lora_data.py
```

Run zero-shot and evaluation on GPU:

```bash
python run_zero_shot.py
python run_evaluation.py
```

Main notebook:

```text
notebooks/task5_expert_surgical_communication_reproducible.ipynb
```

---

## Evaluation

Each model output is scored from 0 to 5:

| Criterion           | Meaning                         |
| ------------------- | ------------------------------- |
| JSON valid          | Correct output format           |
| Hallucination-free  | No unsupported anatomy/tools    |
| Phase safe          | No unsafe phase overclaim       |
| Uncertainty present | Communicates uncertainty        |
| Expert style        | Uses cautious surgical language |

---

## Results

20 held-out test frames:

| Metric              | Zero-shot |   LoRA |
| ------------------- | --------: | -----: |
| JSON valid          |      1.00 |   1.00 |
| Hallucination-free  |      0.00 |   0.00 |
| Phase-safe          |      0.00 |   0.70 |
| Uncertainty-present |      1.00 |   1.00 |
| Expert-style        |      1.00 |   1.00 |
| Average score       |    3.00/5 | 3.70/5 |

LoRA improved phase safety, but hallucination remains a limitation.

---

## Model

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Change in `config.py` to test larger models.

---

## Note

This is a research prototype only. It is not for clinical decision-making.





### **For Task 3**: Simultaneous assessment of instrument tracking precision, anatomical context, and clinical safety grounding.
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
cd task3/data/surgvu
# Download the validation set (includes clips already sampled at 1 FPS)
wget https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip
unzip cat1_test_set_public.zip
# Download the updated labels
wget https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip
unzip surgvu24_labels_updated_v2.zip
# Preprocess data
uv run dataset.py
```

Zero-shot
```bash
cd ../src & uv run baseline_eval.py
```
The srcipt processes your surgical test images (surg_vlm_test.jsonl) using Qwen's specific native vision utilities, capping the image resolution to prevent out-of-memory errors on consumer GPUs. It runs batch inference (processing multiple frames at once for speed) and saves the model's raw, unedited guesses next to the ground truth in a file called zero_shot_results.jsonl.


Fine-tuning:
```bash
uv run train_qlora.py
```
The script includes a specialized Qwen25VLDataCollator that dynamically processes your training images and text into the exact tensor format Qwen requires during training. It then injects LoRA (Low-Rank Adapters) specifically into the attention and MLP layers. This trains only a tiny fraction of the model (~1%), saving massive amounts of compute. Output: It generates the surg_intellect_final_adapter weights, giving the model its **cognitive intelligence layer**.

Evaluation:
```bash
uv run adapter_eval.py
uv run evaluate_metrics.py
# evaluate how the model's performance evolves over the duration of a surgical procedure
uv run temporal_analysis.py
```
**run_adapter_eval.py** evaluates the newly trained **SurgIntellect** model to see what it learned. It operates almost identically to the baseline script, but with one massive difference: it loads the base Qwen2.5-VL-3B-Instruct model and then mathematically merges the new LoRA adapters (PeftModel.from_pretrained) into it. It runs the same test images through this updated brain and saves the output to finetuned_results.jsonl. This generates the data needed that should prove the model has stopped hallucinating and started adopting expert clinical dialogue.

**evaluate_metrics.py** measures clinical safety, by parsing both zero-shot and fine-tuned .jsonl files and calculates three specific metrics:
* Safety Adherence: Did the model use the exact required phrasing ("stable tool-tissue interaction" and "no critical safety violations")?
* Hallucination / Chatter: Base VLMs tend to write paragraphs of dangerous assumptions. This flags any output that is overly long (>200 characters).
* Tool Recall: Did the model correctly mention the specific tools (like needle_driver) listed in the ground truth?

**temporal_analysis.py** evaluates if the model can consistently track the state of the surgery over time, as surgery is not a single static image; it is a long, continuous process. It analyzes the text output of the model and maps specific tools to surgical phases (e.g., if it predicts "needle_driver", the script tags that frame as the "Suturing Phase"). Then it plots these phases continuously over a timeline, generating a visual chart of the surgery's progression.

=> This should show that the pipeline can handle **structured temporal processing** and isn't just making isolated guesses frame-by-frame.
<img width="3600" height="900" alt="temporal_segmentation" src="https://github.com/user-attachments/assets/db6c4de5-e6b5-43c0-adb9-c0c0b99bfbfc" />


