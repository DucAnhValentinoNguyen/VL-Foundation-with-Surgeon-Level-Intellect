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

```text
├──task1/
|    ├── data/ 
|    │   ├── testing/
|    │   ├── training/
|    ├── src/ 
|    │   ├── build_gt_jsonl.py
|    │   ├── eval_phase_predictions.py
|    │   ├── phase_data.py
|    │   ├── phase_model.py
|    │   ├── plots.py
|    │   ├── run_phase_finetuned_inference.py
|    │   ├── train_phase_lora.py
|    │   ├── utils.py
|    ├── results/ 
|    │   ├── eval_finetuned_phase/
|    │   ├── eval_zeroshot_phase/
|    │   ├── visualizations/
|    │   ├── finetuned_predictions.jsonl
|    │   ├── train_history.json
|    │   ├── zeroshot_predictions.jsonl
|    ├── notebook.ipynb
|    ├── requirements.txt
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

## Dataset


