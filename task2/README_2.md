Reproducible prototype for **Task 2: Expert-Level Surgical Communication** in a surgical vision-language model pipeline.

The goal is to test whether a general vision-language model, **Qwen2.5-VL**, can describe laparoscopic cholecystectomy frames in a cautious, annotation-grounded, expert-style format, and whether small LoRA adaptation improves the model's safety and grounding.

---

## Project idea

The full surgical VLM assistant can be separated into five tasks:

1. Tool detection  
2. Phase recognition  
3. Scene graph generation  
4. Temporal understanding  
5. Expert-level surgical communication  

This prototype focuses on **Task 5**.

The input is a laparoscopic surgery image/frame. The output is a structured surgical communication answer with:

- visible instruments
- visible anatomy or tissue
- visible action
- possible surgical phase
- expert surgical description
- uncertainty note

The model is expected to avoid unsupported claims about anatomy, tools, surgical phase, complications, Critical View of Safety, clipping, cutting, or duct division.

---

## Main pipeline

```text
Raw surgical frame + CholecSeg8k annotation
        ↓
Extract visible classes from annotation
        ↓
Generate literature-guided, annotation-grounded teacher answer
        ↓
Prepare Qwen-compatible LoRA instruction data
        ↓
Run Qwen2.5-VL zero-shot
        ↓
Evaluate zero-shot output
        ↓
Fine-tune Qwen2.5-VL with LoRA
        ↓
Run LoRA-Qwen inference
        ↓
Compare zero-shot vs LoRA
```

---

## Dataset

This prototype uses the **sample version of CholecSeg8k** from DatasetNinja.

Expected dataset structure:

```text
surgical_vlm/
├── ds/
│   ├── img/
│   ├── ann/
│   └── meta.json
```

The sample dataset used here contains **963 annotated frames** grouped into **10 video sequences**.

The dataset is split by video ID to reduce leakage between visually similar frames:

```text
Train videos: 0, 1, 2, 3, 4, 5, 6
Validation videos: 7
Test videos: 8, 9
```

Resulting split:

```text
Train: 674 frames
Validation: 146 frames
Test: 143 frames
```

---

## Class grouping

The CholecSeg8k classes are grouped into:

### Instrument classes

```text
grasper
l-hook electrocautery
```

### Anatomy/tissue classes

```text
abdominal wall
blood
connective tissue
cystic duct
fat
gallbladder
gastrointestinal tract
hepatic vein
liver
liver ligament
```

### Ignored class

```text
black background
```

---

## Repository structure

```text
surgical_vlm/
├── config.py
├── README.md
├── requirements.txt
│
├── ds/
│   ├── img/
│   ├── ann/
│   └── meta.json
│
├── notebooks/
│   └── task5_expert_surgical_communication_reproducible.ipynb
│
├── outputs/
│   ├── annotations/
│   ├── lora_data/
│   ├── zero_shot_predictions/
│   ├── lora_predictions/
│   ├── lora_adapter/
│   ├── evaluation/
│   └── figures/
│
├── scripts/
│   ├── __init__.py
│   ├── experiment.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── annotation_builder.py
│   │   └── teacher_labeler.py
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── zero_shot.py
│   │   ├── lora_dataset.py
│   │   ├── lora_train.py
│   │   └── lora_inference.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── visualization.py
│   │
│   └── prompts/
│       ├── __init__.py
│       └── surgical_prompts.py
│
├── run_prepare_annotations.py
├── run_teacher_label.py
├── run_prepare_lora_data.py
├── run_zero_shot.py
└── run_evaluation.py
```

---

## Installation

In Colab or a local GPU environment:

```bash
pip install -r requirements.txt
```

Recommended packages:

```text
torch
torchvision
transformers
accelerate
pillow
qwen-vl-utils
peft
bitsandbytes
tqdm
pandas
matplotlib
```

For Colab, if Qwen2.5-VL has compatibility issues, install the latest Transformers version:

```bash
pip install -q git+https://github.com/huggingface/transformers
```

---

## Reproducible notebook

The main reproducible notebook is:

```text
notebooks/task5_expert_surgical_communication_reproducible.ipynb
```

The notebook is designed to run from Google Colab with the project folder uploaded to Google Drive:

```text
/content/drive/MyDrive/surgical_vlm
```

It performs the full pipeline:

1. Mount Google Drive
2. Set project root
3. Verify dataset
4. Build annotation-grounded samples
5. Build literature-guided teacher labels
6. Prepare LoRA data
7. Run Qwen2.5-VL zero-shot
8. Evaluate zero-shot output
9. Train a small LoRA adapter
10. Run LoRA inference
11. Evaluate LoRA output
12. Compare zero-shot vs LoRA

---

## Running the pipeline manually

### 1. Prepare annotation-grounded samples

```bash
python run_prepare_annotations.py
```

Expected output:

```text
Saved 963 expert annotations
Train samples: 674
Val samples: 146
Test samples: 143
```

### 2. Build teacher labels

```bash
python run_teacher_label.py
```

This creates literature-guided, annotation-grounded expert communication targets.

### 3. Prepare LoRA instruction data

```bash
python run_prepare_lora_data.py
```

Expected outputs:

```text
outputs/lora_data/qwen_lora_train.json
outputs/lora_data/qwen_lora_val.json
outputs/lora_data/qwen_lora_test.json
```

### 4. Run Qwen zero-shot

Run this only in a GPU environment:

```bash
python run_zero_shot.py
```

### 5. Evaluate zero-shot output

```bash
python run_evaluation.py
```

---

## Teacher-labeling strategy

The teacher answer is **not generated by a paid external LLM**.

Instead, this project uses a **literature-guided rule teacher**:

- CholecSeg8k annotation tells what is visible.
- Laparoscopic cholecystectomy knowledge guides safe communication style.
- The generated answer does not invent anatomy, instruments, phases, or complications.

Example principle:

```text
If no grasper is annotated, the answer must not claim grasper-based retraction.
If no l-hook electrocautery is annotated, the answer must not claim electrocautery dissection.
If the frame is a single image, the phase should remain uncertain unless explicitly supported.
```

This creates safer reference answers for LoRA fine-tuning.

---

## Evaluation

The evaluation focuses on communication safety rather than only recognition accuracy.

Each output is scored from 0 to 5:

| Criterion | Meaning | Score |
|---|---|---:|
| JSON valid | Model followed requested JSON format | 1 |
| Hallucination-free | No unsupported instruments/anatomy | 1 |
| Phase safe | No unsafe phase/action overclaim | 1 |
| Uncertainty present | Model communicates uncertainty | 1 |
| Expert style | Uses surgical, cautious, visible-evidence language | 1 |

Maximum score: **5/5**.

---

## Current experiment result

A small experiment was run on **20 held-out test frames**.

### Zero-shot Qwen2.5-VL-3B

| Metric | Value |
|---|---:|
| Samples | 20 |
| JSON valid rate | 1.00 |
| Hallucination-free rate | 0.00 |
| Phase-safe rate | 0.00 |
| Uncertainty-present rate | 1.00 |
| Expert-style rate | 1.00 |
| Average score | 3.00/5 |

### LoRA-adapted Qwen2.5-VL-3B

| Metric | Value |
|---|---:|
| Samples | 20 |
| JSON valid rate | 1.00 |
| Hallucination-free rate | 0.00 |
| Phase-safe rate | 0.70 |
| Uncertainty-present rate | 1.00 |
| Expert-style rate | 1.00 |
| Average score | 3.70/5 |

---

## Interpretation

The zero-shot model follows the requested JSON format and produces uncertainty-aware surgical language. However, it often hallucinates instruments, anatomy, actions, or surgical phases that are not supported by the CholecSeg8k annotation.

After a small LoRA fine-tuning run, phase safety improves substantially, increasing the average evaluation score from **3.00/5** to **3.70/5**. This suggests that LoRA helps the model learn safer surgical communication patterns.

The hallucination-free rate remains low in this prototype, which indicates that stronger visual grounding is still needed. Future improvements could include:

- more training samples
- stricter negative examples
- stronger label-to-output constraints
- segmentation-aware visual grounding
- task-specific tool/anatomy detector modules
- more robust evaluation with expert review

---

## Model choice

The current prototype uses:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

This model is selected because it is practical for Colab GPU experiments.

For stronger experiments, the model can be changed in `config.py`:

```python
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
```

The 7B model may require a stronger GPU.

---

## Limitations

This is a prototype, not a clinical system.

Important limitations:

- The dataset is a sample subset, not the full CholecSeg8k dataset.
- The teacher labels are rule-based and annotation-grounded, not surgeon-reviewed.
- The model is evaluated on a small test subset.
- Single-frame inputs cannot reliably determine full surgical phase or temporal context.
- The current evaluation is automatic and approximate.
- The model should not be used for clinical decision-making.

---

## Future work

Potential extensions:

1. Train on the full CholecSeg8k dataset.
2. Add temporal context using frame sequences.
3. Add scene graph generation for tool–tissue relationships.
4. Use segmentation masks as visual grounding.
5. Add expert surgeon review for teacher labels and evaluation.
6. Fine-tune Qwen2.5-VL-7B or larger models.
7. Compare against medical VLMs or surgical VLM baselines.
8. Integrate retrieval-augmented surgical knowledge for safer communication.

---

## Safety note

This project is for research and prototyping only. It does not provide medical advice and must not be used as a clinical decision-support system without expert validation, regulatory review, and rigorous safety testing.

