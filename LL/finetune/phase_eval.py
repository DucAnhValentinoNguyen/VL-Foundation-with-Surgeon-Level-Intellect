import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from zeroshot.phase_eval import phase_map,segments_to_frame_labels,save_confusion_matrix


pred_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetuned_phase_predictions.jsonl"
output_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/eval_finetuned_phase_task"


os.makedirs(output_dir, exist_ok=True)



with open(pred_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

all_gt = []
all_pred = []

for row in rows:
    gt = row["ground_truth"]
    pred = row["prediction"]

    gt_segments = gt["phase_segments"]
    pred_segments = pred["phase_segments"]

    gt_seq = segments_to_frame_labels(gt_segments)
    pred_seq = segments_to_frame_labels(pred_segments)

    n = min(len(gt_seq), len(pred_seq))
    if n == 0:
        continue

    gt_seq = gt_seq[:n]
    pred_seq = pred_seq[:n]

    for g, p in zip(gt_seq, pred_seq):
        if g != -1 and p != -1:
            all_gt.append(g)
            all_pred.append(p)

phase_accuracy = accuracy_score(all_gt, all_pred) if len(all_gt) > 0 else 0.0
phase_macro_f1 = f1_score(all_gt, all_pred, average="macro", zero_division=0) if len(all_gt) > 0 else 0.0

summary = {
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
        "Finetuned Phase Confusion Matrix",
        os.path.join(output_dir, "phase_confusion_matrix.png")
    )

print("saved to:", output_dir)