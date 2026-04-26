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
Please read our report

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

## ⚙️ Setup and Execution
**For Task 3**: Simultaneous assessment of instrument tracking precision, anatomical context, and clinical safety grounding.
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


