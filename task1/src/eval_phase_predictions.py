# Final evaluation utilities for phase recognition:
# - validate the unified prediction format,
# - compute JSON parse success rate and schema validity rate,
# - compute frame-wise phase accuracy and macro F1,
# - save the phase confusion matrix and summary metrics.

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import json
import numpy as np
from pathlib import Path
from utils import safe_parse_json
from phase_data import load_jsonl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ZEROSHOT_PRED = DATA_DIR / "results" / "zeroshot_predictions.jsonl"
FINETUNE_PRED = DATA_DIR / "results" / "finetuned_predictions.jsonl"
ZEROSHOT_OUT = BASE_DIR / "results" / "eval_zeroshot_phase"
FINETUNE_OUT = BASE_DIR / "results" / "eval_finetuned_phase"


phase_map = {
    0: "Preparation",
    1: "Calot triangle dissection",
    2: "Clipping and cutting",
    3: "Gallbladder dissection",
    4: "Gallbladder retraction",
    5: "Cleaning and coagulation",
    6: "Gallbladder packaging",
}


# Check whether a prediction follows the unified frame-wise phase schema.
def schema_valid_phase_prediction(obj, num_phase_classes=7):
    if not isinstance(obj, dict):
        return False

    if "frame_indices" not in obj or "phase_labels" not in obj:
        return False

    frame_indices = obj["frame_indices"]
    phase_labels = obj["phase_labels"]

    if not isinstance(frame_indices, list) or not isinstance(phase_labels, list):
        return False

    if len(frame_indices) != len(phase_labels):
        return False

    for x in frame_indices:
        if not isinstance(x, int):
            return False

    for x in phase_labels:
        if not isinstance(x, int):
            return False
        if x != -1 and not (0 <= x < num_phase_classes):
            return False

    return True


# Save a confusion matrix figure for phase classification results.
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


# Evaluate a saved prediction JSONL file and export summary metrics and confusion matrix.
def evaluate_phase_jsonl(pred_path, output_dir, num_phase_classes=7):
    os.makedirs(output_dir, exist_ok=True)

    rows = load_jsonl(pred_path)

    num_samples = len(rows)
    parse_ok = 0
    schema_ok = 0

    all_gt = []
    all_pred = []

    for row in rows:
        gt = row.get("ground_truth", None)
        pred = row.get("prediction", None)
        raw_pred = row.get("raw_prediction", None)

        if raw_pred is None:
            parse_ok += 1
        else:
            if safe_parse_json(raw_pred) is not None:
                parse_ok += 1

        if schema_valid_phase_prediction(pred, num_phase_classes=num_phase_classes):
            schema_ok += 1

        if not schema_valid_phase_prediction(gt, num_phase_classes=num_phase_classes):
            continue

        if not schema_valid_phase_prediction(pred, num_phase_classes=num_phase_classes):
            continue

        gt_map = {
            int(fid): int(lbl)
            for fid, lbl in zip(gt["frame_indices"], gt["phase_labels"])
        }
        pred_map = {
            int(fid): int(lbl)
            for fid, lbl in zip(pred["frame_indices"], pred["phase_labels"])
        }

        common_frame_indices = sorted(set(gt_map.keys()) & set(pred_map.keys()))

        for fid in common_frame_indices:
            g = gt_map[fid]
            p = pred_map[fid]
            if g != -1 and p != -1:
                all_gt.append(g)
                all_pred.append(p)

    json_parse_success_rate = parse_ok / num_samples if num_samples > 0 else 0.0
    schema_validity_rate = schema_ok / num_samples if num_samples > 0 else 0.0
    framewise_phase_accuracy = accuracy_score(all_gt, all_pred) if len(all_gt) > 0 else 0.0
    framewise_phase_macro_f1 = f1_score(all_gt, all_pred, average="macro", zero_division=0) if len(all_gt) > 0 else 0.0

    summary = {
        "json_parse_success_rate": json_parse_success_rate,
        "schema_validity_rate": schema_validity_rate,
        "framewise_phase_accuracy": framewise_phase_accuracy,
        "framewise_phase_macro_f1": framewise_phase_macro_f1,
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
            "Phase Confusion Matrix",
            os.path.join(output_dir, "phase_confusion_matrix.png")
        )

    print("saved to:", output_dir)


if __name__ == "__main__":

    evaluate_phase_jsonl(
        pred_path=ZEROSHOT_PRED,
        output_dir=ZEROSHOT_OUT,
        num_phase_classes=7
    )
    evaluate_phase_jsonl(
        pred_path=FINETUNE_PRED,
        output_dir=FINETUNE_OUT,
        num_phase_classes=7
    )