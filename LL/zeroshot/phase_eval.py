import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


pred_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/zeroshot_phase_predictions.jsonl"
output_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/eval_phase_task"


os.makedirs(output_dir, exist_ok=True)

phase_map = {
    0: "Preparation",
    1: "Calot triangle dissection",
    2: "Clipping and cutting",
    3: "Gallbladder dissection",
    4: "Gallbladder retraction",
    5: "Cleaning and coagulation",
    6: "Gallbladder packaging",
}

def extract_json_block(text):
    if not isinstance(text, str):
        return text
    text = text.strip()
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
    return text

def safe_parse_json(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    try:
        return json.loads(extract_json_block(x))
    except Exception:
        return None

def schema_valid_phase(obj):
    if not isinstance(obj, dict):
        return False
    segs = obj.get("phase_segments")
    if not isinstance(segs, list):
        return False
    for seg in segs:
        if not isinstance(seg, dict):
            return False
        required = {"start_frame", "end_frame", "phase"}
        if not required.issubset(seg.keys()):
            return False
    return True

def segments_to_frame_labels(segments):

    if not isinstance(segments, list) or len(segments) == 0:
        return []

    max_end = max(int(seg["end_frame"]) for seg in segments)
    arr = [-1] * (max_end + 1)

    for seg in segments:
        s = int(seg["start_frame"])
        e = int(seg["end_frame"])
        p = int(seg["phase"])
        s = max(s, 0)
        e = max(e, s)
        for i in range(s, e + 1):
            if i < len(arr):
                arr[i] = p
    return arr

def save_confusion_matrix(cm, labels, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

with open(pred_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

num_samples = len(rows)
parse_ok = 0
schema_ok = 0

all_gt = []
all_pred = []

for row in rows:
    gt = safe_parse_json(row.get("ground_truth"))
    pred = safe_parse_json(row.get("prediction"))

    if pred is not None:
        parse_ok += 1
    if schema_valid_phase(pred):
        schema_ok += 1

    if gt is None or pred is None:
        continue
    if not schema_valid_phase(gt) or not schema_valid_phase(pred):
        continue

    gt_seq = segments_to_frame_labels(gt["phase_segments"])
    pred_seq = segments_to_frame_labels(pred["phase_segments"])

    n = min(len(gt_seq), len(pred_seq))
    if n == 0:
        continue

    gt_seq = gt_seq[:n]
    pred_seq = pred_seq[:n]

    for g, p in zip(gt_seq, pred_seq):
        if g != -1 and p != -1:
            all_gt.append(g)
            all_pred.append(p)

json_parse_success_rate = parse_ok / num_samples if num_samples > 0 else 0.0
schema_validity_rate = schema_ok / num_samples if num_samples > 0 else 0.0
phase_accuracy = accuracy_score(all_gt, all_pred) if len(all_gt) > 0 else 0.0
phase_macro_f1 = f1_score(all_gt, all_pred, average="macro", zero_division=0) if len(all_gt) > 0 else 0.0

summary = {
    "json_parse_success_rate": json_parse_success_rate,
    "schema_validity_rate": schema_validity_rate,
    "framewise_phase_accuracy": phase_accuracy,
    "framewise_phase_macro_f1": phase_macro_f1,
}

with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(summary)

if len(all_gt) > 0:
    labels = sorted(set(all_gt) | set(all_pred))
    cm = confusion_matrix(all_gt, all_pred, labels=labels)
    label_names = [phase_map.get(x, f"phase_{x}") for x in labels]

    save_confusion_matrix(
        cm,
        label_names,
        "Phase Task Confusion Matrix",
        os.path.join(output_dir, "phase_confusion_matrix.png")
    )

print("saved to:", output_dir)