import os
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

pred_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/zeroshot_anchor_predictions.jsonl"
output_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/eval_anchor_task"
iou_threshold = 0.5


os.makedirs(output_dir, exist_ok=True)

instrument_map = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
    6: "specimen-bag",
}

operator_map = {
    0: "null",
    1: "main-surgeon-left-hand (MSLH)",
    2: "assistant-surgeon-right-hand (ASRH)",
    3: "main-surgeon-right-hand (MSRH)",
}

phase_map = {
    0: "Preparation",
    1: "Calot triangle dissection",
    2: "Clipping and cutting",
    3: "Gallbladder dissection",
    4: "Gallbladder retraction",
    5: "Cleaning and coagulation",
    6: "Gallbladder packaging",
}


# -------------------------
# Utils
# -------------------------
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


def normalize_bbox(b):

    if b is None:
        return None

    # [[a,b,c,d]] -> [a,b,c,d]
    if isinstance(b, list) and len(b) == 1 and isinstance(b[0], list):
        b = b[0]

    if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        return [float(x) for x in b]

    return None


def normalize_track(x):
    return x if isinstance(x, int) else None


def normalize_object(obj):
    if not isinstance(obj, dict):
        return {
            "instrument": None,
            "tool_bbox": None,
            "operator": None,
            "intraoperative_track": None,
            "intracorporeal_track": None,
        }

    instrument = obj.get("instrument")
    operator = obj.get("operator")

    return {
        "instrument": instrument if isinstance(instrument, int) else None,
        "tool_bbox": normalize_bbox(obj.get("tool_bbox")),
        "operator": operator if isinstance(operator, int) else None,
        "intraoperative_track": normalize_track(obj.get("intraoperative_track")),
        "intracorporeal_track": normalize_track(obj.get("intracorporeal_track")),
    }


def normalize_frame(frame):
    if not isinstance(frame, dict):
        return {"frame_index": None, "phase": None, "objects": []}

    frame_index = frame.get("frame_index")
    phase = frame.get("phase")
    objects = frame.get("objects", [])

    if not isinstance(objects, list):
        objects = []

    return {
        "frame_index": frame_index if isinstance(frame_index, int) else None,
        "phase": phase if isinstance(phase, int) else None,
        "objects": [normalize_object(o) for o in objects],
    }


def get_frames(obj):
    if not isinstance(obj, dict):
        return []
    frames = obj.get("frames", [])
    if not isinstance(frames, list):
        return []
    return [normalize_frame(f) for f in frames]


def schema_valid(obj):
    if not isinstance(obj, dict):
        return False

    frames = obj.get("frames")
    if not isinstance(frames, list):
        return False

    for frame in frames:
        if not isinstance(frame, dict):
            return False
        if "frame_index" not in frame or "phase" not in frame or "objects" not in frame:
            return False
        if not isinstance(frame["objects"], list):
            return False

        for ob in frame["objects"]:
            if not isinstance(ob, dict):
                return False
            required = {
                "instrument",
                "tool_bbox",
                "operator",
                "intraoperative_track",
                "intracorporeal_track",
            }
            if not required.issubset(ob.keys()):
                return False

    return True


def valid_bbox(b):
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(isinstance(v, (int, float)) for v in b)
    )


def bbox_iou_xywh(box1, box2):
    if not valid_bbox(box1) or not valid_bbox(box2):
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def hungarian_match(gt_objects, pred_objects, iou_thr=0.5):
    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    if n_gt == 0 and n_pred == 0:
        return [], [], []
    if n_gt == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_gt)), []

    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, g in enumerate(gt_objects):
        for j, p in enumerate(pred_objects):
            iou_mat[i, j] = bbox_iou_xywh(g.get("tool_bbox"), p.get("tool_bbox"))

    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        iou = float(iou_mat[r, c])
        if iou >= iou_thr:
            matches.append((r, c, iou))
            matched_gt.add(r)
            matched_pred.add(c)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def save_confusion_matrix(cm, labels, title, out_path):
    fig_w = max(8, 0.8 * len(labels))
    fig_h = max(6, 0.8 * len(labels))
    plt.figure(figsize=(fig_w, fig_h))
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


def save_bar_chart(labels, values, title, ylabel, out_path):
    plt.figure(figsize=(max(10, 0.9 * len(labels)), 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Load predictions
# -------------------------
with open(pred_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

num_samples = len(rows)

# -------------------------
# A. Format
# -------------------------
parse_ok = 0
schema_ok = 0

# -------------------------
# B. Phase
# -------------------------
phase_gt_all = []
phase_pred_all = []

# -------------------------
# C. Detection / localization
# -------------------------
tp = 0
fp = 0
fn = 0
iou_values = []
per_instrument_iou = defaultdict(list)

# -------------------------
# D. Matched-object classification
# -------------------------
instr_gt_all = []
instr_pred_all = []

op_gt_all = []
op_pred_all = []

intraop_gt_all = []
intraop_pred_all = []

intracorp_gt_all = []
intracorp_pred_all = []


# -------------------------
# Main loop
# -------------------------
for row in rows:
    gt = safe_parse_json(row.get("ground_truth"))
    pred = safe_parse_json(row.get("prediction"))

    gt_valid = schema_valid(gt)
    pred_valid = schema_valid(pred)

    if pred is not None:
        parse_ok += 1
    if pred_valid:
        schema_ok += 1

    if (not gt_valid) or (not pred_valid):
        continue

    gt_frames = get_frames(gt)
    pred_frames = get_frames(pred)

    gt_map = {f["frame_index"]: f for f in gt_frames if f["frame_index"] is not None}
    pred_map = {f["frame_index"]: f for f in pred_frames if f["frame_index"] is not None}

    common_frame_indices = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    gt_only_indices = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    pred_only_indices = sorted(set(pred_map.keys()) - set(gt_map.keys()))

    for fid in common_frame_indices:
        gt_f = gt_map[fid]
        pred_f = pred_map[fid]

        if gt_f["phase"] is not None and pred_f["phase"] is not None:
            phase_gt_all.append(gt_f["phase"])
            phase_pred_all.append(pred_f["phase"])

        gt_objs = gt_f["objects"]
        pred_objs = pred_f["objects"]

        matches, unmatched_gt, unmatched_pred = hungarian_match(
            gt_objs, pred_objs, iou_thr=iou_threshold
        )

        tp += len(matches)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)

        for gi, pi, iou in matches:
            g = gt_objs[gi]
            p = pred_objs[pi]

            iou_values.append(iou)

            if g["instrument"] is not None:
                per_instrument_iou[g["instrument"]].append(iou)

            if g["instrument"] is not None and p["instrument"] is not None:
                instr_gt_all.append(g["instrument"])
                instr_pred_all.append(p["instrument"])

            if g["operator"] is not None and p["operator"] is not None:
                op_gt_all.append(g["operator"])
                op_pred_all.append(p["operator"])

            if g["intraoperative_track"] is not None and p["intraoperative_track"] is not None:
                intraop_gt_all.append(g["intraoperative_track"])
                intraop_pred_all.append(p["intraoperative_track"])

            if g["intracorporeal_track"] is not None and p["intracorporeal_track"] is not None:
                intracorp_gt_all.append(g["intracorporeal_track"])
                intracorp_pred_all.append(p["intracorporeal_track"])

    for fid in gt_only_indices:
        fn += len(gt_map[fid]["objects"])

    for fid in pred_only_indices:
        fp += len(pred_map[fid]["objects"])


# -------------------------
# Compute metrics
# -------------------------
json_parse_success_rate = parse_ok / num_samples if num_samples > 0 else 0.0
schema_validity_rate = schema_ok / num_samples if num_samples > 0 else 0.0

phase_accuracy = accuracy_score(phase_gt_all, phase_pred_all) if len(phase_gt_all) > 0 else 0.0
phase_macro_f1 = f1_score(phase_gt_all, phase_pred_all, average="macro", zero_division=0) if len(phase_gt_all) > 0 else 0.0

detection_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
detection_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
detection_f1 = (
    2 * detection_precision * detection_recall / (detection_precision + detection_recall)
    if (detection_precision + detection_recall) > 0 else 0.0
)

bbox_mean_iou = float(np.mean(iou_values)) if len(iou_values) > 0 else 0.0

instrument_accuracy = accuracy_score(instr_gt_all, instr_pred_all) if len(instr_gt_all) > 0 else 0.0
instrument_macro_f1 = f1_score(instr_gt_all, instr_pred_all, average="macro", zero_division=0) if len(instr_gt_all) > 0 else 0.0
instrument_report = classification_report(
    instr_gt_all,
    instr_pred_all,
    labels=sorted(instrument_map.keys()),
    target_names=[instrument_map[k] for k in sorted(instrument_map.keys())],
    zero_division=0,
    output_dict=False
) if len(instr_gt_all) > 0 else "No matched objects."

operator_accuracy = accuracy_score(op_gt_all, op_pred_all) if len(op_gt_all) > 0 else 0.0
operator_macro_f1 = f1_score(op_gt_all, op_pred_all, average="macro", zero_division=0) if len(op_gt_all) > 0 else 0.0
operator_report = classification_report(
    op_gt_all,
    op_pred_all,
    labels=sorted(operator_map.keys()),
    target_names=[operator_map[k] for k in sorted(operator_map.keys())],
    zero_division=0,
    output_dict=False
) if len(op_gt_all) > 0 else "No matched objects."

intraoperative_track_exact_match_accuracy = accuracy_score(intraop_gt_all, intraop_pred_all) if len(intraop_gt_all) > 0 else 0.0
intraoperative_track_macro_f1 = f1_score(intraop_gt_all, intraop_pred_all, average="macro", zero_division=0) if len(intraop_gt_all) > 0 else 0.0
intraoperative_track_micro_f1 = f1_score(intraop_gt_all, intraop_pred_all, average="micro", zero_division=0) if len(intraop_gt_all) > 0 else 0.0

intracorporeal_track_exact_match_accuracy = accuracy_score(intracorp_gt_all, intracorp_pred_all) if len(intracorp_gt_all) > 0 else 0.0
intracorporeal_track_macro_f1 = f1_score(intracorp_gt_all, intracorp_pred_all, average="macro", zero_division=0) if len(intracorp_gt_all) > 0 else 0.0
intracorporeal_track_micro_f1 = f1_score(intracorp_gt_all, intracorp_pred_all, average="micro", zero_division=0) if len(intracorp_gt_all) > 0 else 0.0


# -------------------------
# Save summary
# -------------------------
summary = {
    "A_format": {
        "json_parse_success_rate": json_parse_success_rate,
        "schema_validity_rate": schema_validity_rate,
    },
    "B_phase": {
        "accuracy": phase_accuracy,
        "macro_f1": phase_macro_f1,
    },
    "C_detection_localization": {
        "iou_threshold_for_matching": iou_threshold,
        "precision": detection_precision,
        "recall": detection_recall,
        "f1": detection_f1,
        "bbox_mean_iou": bbox_mean_iou,
    },
    "D_matched_object_classification": {
        "instrument": {
            "accuracy": instrument_accuracy,
            "macro_f1": instrument_macro_f1,
        },
        "operator": {
            "accuracy": operator_accuracy,
            "macro_f1": operator_macro_f1,
        },
        "intraoperative_track": {
            "exact_match_accuracy": intraoperative_track_exact_match_accuracy,
            "macro_f1": intraoperative_track_macro_f1,
            "micro_f1": intraoperative_track_micro_f1,
        },
        "intracorporeal_track": {
            "exact_match_accuracy": intracorporeal_track_exact_match_accuracy,
            "macro_f1": intracorporeal_track_macro_f1,
            "micro_f1": intracorporeal_track_micro_f1,
        },
    },
}

with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_dir, "instrument_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(instrument_report)

with open(os.path.join(output_dir, "operator_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(operator_report)

if len(phase_gt_all) > 0:
    labels = sorted(set(phase_gt_all) | set(phase_pred_all))
    cm = confusion_matrix(phase_gt_all, phase_pred_all, labels=labels)
    label_names = [phase_map.get(x, f"phase_{x}") for x in labels]

    save_confusion_matrix(
        cm,
        label_names,
        "Anchor Task Phase Confusion Matrix",
        os.path.join(output_dir, "phase_confusion_matrix.png")
    )

inst_ids_sorted = sorted(per_instrument_iou.keys())
inst_names = [instrument_map.get(i, f"instrument_{i}") for i in inst_ids_sorted]
inst_mean_ious = [
    float(np.mean(per_instrument_iou[i])) if len(per_instrument_iou[i]) > 0 else 0.0
    for i in inst_ids_sorted
]

if len(inst_ids_sorted) > 0:
    save_bar_chart(
        inst_names,
        inst_mean_ious,
        "BBox Mean IoU by Instrument Class",
        "Mean IoU",
        os.path.join(output_dir, "bbox_iou_by_instrument.png")
    )

print("=" * 80)
print("A. FORMAT")
print("=" * 80)
print("JSON parse success rate:", round(json_parse_success_rate, 4))
print("Schema validity rate:", round(schema_validity_rate, 4))

print("\n" + "=" * 80)
print("B. PHASE")
print("=" * 80)
print("Accuracy:", round(phase_accuracy, 4))
print("Macro F1:", round(phase_macro_f1, 4))

print("\n" + "=" * 80)
print("C. DETECTION / LOCALIZATION")
print("=" * 80)
print("Hungarian matching IoU threshold:", iou_threshold)
print("Precision:", round(detection_precision, 4))
print("Recall:", round(detection_recall, 4))
print("F1:", round(detection_f1, 4))
print("BBox mean IoU:", round(bbox_mean_iou, 4))

print("\n" + "=" * 80)
print("D. MATCHED-OBJECT CLASSIFICATION")
print("=" * 80)

print("\nInstrument")
print("Accuracy:", round(instrument_accuracy, 4))
print("Macro F1:", round(instrument_macro_f1, 4))
print(instrument_report)

print("\nOperator")
print("Accuracy:", round(operator_accuracy, 4))
print("Macro F1:", round(operator_macro_f1, 4))
print(operator_report)

print("\nIntraoperative Track")
print("Exact match accuracy:", round(intraoperative_track_exact_match_accuracy, 4))
print("Macro F1:", round(intraoperative_track_macro_f1, 4))
print("Micro F1:", round(intraoperative_track_micro_f1, 4))

print("\nIntracorporeal Track")
print("Exact match accuracy:", round(intracorporeal_track_exact_match_accuracy, 4))
print("Macro F1:", round(intracorporeal_track_macro_f1, 4))
print("Micro F1:", round(intracorporeal_track_micro_f1, 4))

print("\nSaved to:", output_dir)
print("- summary_metrics.json")
print("- instrument_classification_report.txt")
print("- operator_classification_report.txt")
print("- phase_confusion_matrix.png")
print("- bbox_iou_by_instrument.png")