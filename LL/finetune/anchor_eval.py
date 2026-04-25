import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from LL.zeroshot.anchor_eval import phase_map, instrument_map, operator_map, save_confusion_matrix, save_bar_chart


pred_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetuned_anchor_predictions.jsonl"
output_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/eval_finetuned_anchor_task"
iou_threshold = 0.5

os.makedirs(output_dir, exist_ok=True)


# -------------------------
# Utils
# -------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def bbox_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou_xywh(box1, box2):
    if box1 is None or box2 is None:
        return 0.0

    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(box1)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(box2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_area_xyxy([ax1, ay1, ax2, ay2])
    area_b = box_area_xyxy([bx1, by1, bx2, by2])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union




# -------------------------
# Matching
# -------------------------


def hungarian_match_by_iou(gt_objects, pred_objects, iou_threshold=0.5):
    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    if n_gt == 0 and n_pred == 0:
        return [], [], []
    if n_gt == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_gt)), []

    cost = np.zeros((n_gt, n_pred), dtype=np.float32)
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)

    for i, gt in enumerate(gt_objects):
        for j, pred in enumerate(pred_objects):
            iou = bbox_iou_xywh(gt["tool_bbox"], pred["tool_bbox"])
            iou_mat[i, j] = iou
            cost[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        iou = float(iou_mat[r, c])
        if iou >= iou_threshold:
            matches.append((r, c, iou))
            matched_gt.add(r)
            matched_pred.add(c)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


# -------------------------
# Main eval
# -------------------------
rows = load_jsonl(pred_path)

# B. Phase
phase_gt_all = []
phase_pred_all = []

# C. Detection
tp = 0
fp = 0
fn = 0
matched_ious = []
per_instr_ious = {k: [] for k in instrument_map.keys()}

# D. Matched-object classification
instr_gt_all = []
instr_pred_all = []

op_gt_all = []
op_pred_all = []

intraop_gt_all = []
intraop_pred_all = []

intracorp_gt_all = []
intracorp_pred_all = []

for row in rows:
    gt = row["ground_truth"]
    pred = row["prediction"]

    gt_frames = gt["frames"]
    pred_frames = pred["frames"]

    n = min(len(gt_frames), len(pred_frames))

    for i in range(n):
        gt_frame = gt_frames[i]
        pred_frame = pred_frames[i]

        # Phase
        phase_gt_all.append(int(gt_frame["phase"]))
        phase_pred_all.append(int(pred_frame["phase"]))

        # Detection / matching
        gt_objects = gt_frame["objects"]
        pred_objects = pred_frame["objects"]

        matches, unmatched_gt, unmatched_pred = hungarian_match_by_iou(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold
        )

        tp += len(matches)
        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

        for gt_idx, pred_idx, iou in matches:
            matched_ious.append(iou)

            gt_obj = gt_objects[gt_idx]
            pred_obj = pred_objects[pred_idx]

            gt_instr = int(gt_obj["instrument"])
            pred_instr = int(pred_obj["instrument"])

            if gt_instr in per_instr_ious:
                per_instr_ious[gt_instr].append(iou)

            instr_gt_all.append(gt_instr)
            instr_pred_all.append(pred_instr)

            op_gt_all.append(int(gt_obj["operator"]))
            op_pred_all.append(int(pred_obj["operator"]))

            intraop_gt_all.append(int(gt_obj["intraoperative_track"]))
            intraop_pred_all.append(int(pred_obj["intraoperative_track"]))

            intracorp_gt_all.append(int(gt_obj["intracorporeal_track"]))
            intracorp_pred_all.append(int(pred_obj["intracorporeal_track"]))


# -------------------------
# B. Phase metrics
# -------------------------
phase_accuracy = accuracy_score(phase_gt_all, phase_pred_all) if len(phase_gt_all) > 0 else 0.0
phase_macro_f1 = f1_score(phase_gt_all, phase_pred_all, average="macro", zero_division=0) if len(phase_gt_all) > 0 else 0.0

phase_labels = list(range(7))
phase_cm = confusion_matrix(phase_gt_all, phase_pred_all, labels=phase_labels) if len(phase_gt_all) > 0 else np.zeros((7, 7), dtype=int)

save_confusion_matrix(
    phase_cm,
    [phase_map[x] for x in phase_labels],
    "Task B Phase Confusion Matrix",
    os.path.join(output_dir, "phase_confusion_matrix.png")
)

# -------------------------
# C. Detection / localization metrics
# -------------------------
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_det = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
bbox_mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0

per_instr_mean_iou = {}
for k, name in instrument_map.items():
    vals = per_instr_ious[k]
    per_instr_mean_iou[name] = float(np.mean(vals)) if len(vals) > 0 else 0.0

save_bar_chart(
    list(per_instr_mean_iou.keys()),
    list(per_instr_mean_iou.values()),
    "BBox IoU by Instrument",
    "Mean IoU",
    os.path.join(output_dir, "bbox_iou_by_instrument.png")
)

# -------------------------
# D. Matched-object classification metrics
# -------------------------
# Instrument
instrument_accuracy = accuracy_score(instr_gt_all, instr_pred_all) if len(instr_gt_all) > 0 else 0.0
instrument_report = classification_report(
    instr_gt_all,
    instr_pred_all,
    labels=list(instrument_map.keys()),
    target_names=[instrument_map[x] for x in instrument_map.keys()],
    zero_division=0,
    digits=4
) if len(instr_gt_all) > 0 else "No matched objects."

# Operator
operator_accuracy = accuracy_score(op_gt_all, op_pred_all) if len(op_gt_all) > 0 else 0.0
operator_report = classification_report(
    op_gt_all,
    op_pred_all,
    labels=list(operator_map.keys()),
    target_names=[operator_map[x] for x in operator_map.keys()],
    zero_division=0,
    digits=4
) if len(op_gt_all) > 0 else "No matched objects."

# Intraoperative track
intraop_exact_acc = accuracy_score(intraop_gt_all, intraop_pred_all) if len(intraop_gt_all) > 0 else 0.0
intraop_macro_f1 = f1_score(intraop_gt_all, intraop_pred_all, average="macro", zero_division=0) if len(intraop_gt_all) > 0 else 0.0
intraop_micro_f1 = f1_score(intraop_gt_all, intraop_pred_all, average="micro", zero_division=0) if len(intraop_gt_all) > 0 else 0.0

# Intracorporeal track
intracorp_exact_acc = accuracy_score(intracorp_gt_all, intracorp_pred_all) if len(intracorp_gt_all) > 0 else 0.0
intracorp_macro_f1 = f1_score(intracorp_gt_all, intracorp_pred_all, average="macro", zero_division=0) if len(intracorp_gt_all) > 0 else 0.0
intracorp_micro_f1 = f1_score(intracorp_gt_all, intracorp_pred_all, average="micro", zero_division=0) if len(intracorp_gt_all) > 0 else 0.0


# -------------------------
# Save summary
# -------------------------
summary = {
    "phase": {
        "accuracy": phase_accuracy,
        "macro_f1": phase_macro_f1,
    },
    "detection_localization": {
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1_det,
        "bbox_mean_iou": bbox_mean_iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_instrument_mean_iou": per_instr_mean_iou,
    },
    "matched_object_classification": {
        "instrument": {
            "accuracy": instrument_accuracy,
        },
        "operator": {
            "accuracy": operator_accuracy,
        },
        "intraoperative_track": {
            "exact_match_accuracy": intraop_exact_acc,
            "macro_f1": intraop_macro_f1,
            "micro_f1": intraop_micro_f1,
        },
        "intracorporeal_track": {
            "exact_match_accuracy": intracorp_exact_acc,
            "macro_f1": intracorp_macro_f1,
            "micro_f1": intracorp_micro_f1,
        },
    },
}

with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_dir, "instrument_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(instrument_report)

with open(os.path.join(output_dir, "operator_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(operator_report)


# -------------------------
# Pretty print
# -------------------------
print("TASK B\n")

print("================================================================================")
print("B. PHASE")
print("================================================================================")
print(f"Accuracy: {phase_accuracy:.4f}")
print(f"Macro F1: {phase_macro_f1:.4f}")

print("\n================================================================================")
print("C. DETECTION / LOCALIZATION")
print("================================================================================")
print(f"Hungarian matching IoU threshold: {iou_threshold}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1_det:.4f}")
print(f"BBox mean IoU: {bbox_mean_iou:.4f}")

print("\nPer-instrument mean IoU:")
for name, val in per_instr_mean_iou.items():
    print(f"  {name}: {val:.4f}")

print("\n================================================================================")
print("D. MATCHED-OBJECT CLASSIFICATION")
print("================================================================================")

print("\nInstrument")
print(f"Accuracy: {instrument_accuracy:.4f}")
print(instrument_report)

print("\nOperator")
print(f"Accuracy: {operator_accuracy:.4f}")
print(operator_report)

print("\nIntraoperative Track")
print(f"Exact match accuracy: {intraop_exact_acc:.4f}")
print(f"Macro F1: {intraop_macro_f1:.4f}")
print(f"Micro F1: {intraop_micro_f1:.4f}")

print("\nIntracorporeal Track")
print(f"Exact match accuracy: {intracorp_exact_acc:.4f}")
print(f"Macro F1: {intracorp_macro_f1:.4f}")
print(f"Micro F1: {intracorp_micro_f1:.4f}")

print("\nSaved to:", output_dir)
print("- summary_metrics.json")
print("- instrument_classification_report.txt")
print("- operator_classification_report.txt")
print("- phase_confusion_matrix.png")
print("- bbox_iou_by_instrument.png")