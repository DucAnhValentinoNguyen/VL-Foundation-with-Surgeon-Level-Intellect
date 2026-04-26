# Final inference setup:
# load the test JSONL, the trained checkpoint, and define
# where the structured phase predictions will be saved.

import os
import json
from pathlib import Path
import cv2
import time
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from phase_data import TaskAPhaseDataset, collate_fn
from phase_model import TaskAQwen25VL


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEST_JSONL = DATA_DIR / "testing" / "gt_10s.jsonl"
CKPT_PATH = BASE_DIR / "outputs"/ "qwen25vl_frozen.pt"
PRED_OUT = BASE_DIR / "outputs" / "finetuned_predictions.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

num_phase_classes = 7
device = "cuda" if torch.cuda.is_available() else "cpu"

max_frames_per_video = 90


# Run final phase inference on the test split using the trained model,
# and save predictions in the unified evaluation format.

@torch.no_grad()
def run_inference():

    dataset = TaskAPhaseDataset(
        TEST_JSONL,
        max_frames_per_video=max_frames_per_video,
        num_phase_classes=num_phase_classes,
        validate=False,
        random_offset_sec=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TaskAQwen25VL(
        model_name=MODEL_NAME,
        num_phase_classes=num_phase_classes,
        use_lora=False,
        freeze_backbone=True,
        backbone_dtype=torch.float32,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    rows = []

    for i, sample in enumerate(loader):
        print(f"running phase inference {i+1}/{len(dataset)}")

        phase_logits = model(sample["frames"])
        pred_phase_ids = phase_logits.argmax(dim=-1)[0].detach().cpu().tolist()

        out_row = {
            "video": sample["video"],
            "clip_json_path": sample.get("clip_json_path", None),
            "ground_truth": {
                "frame_indices": sample["frame_indices"],
                "phase_labels": sample["phase_labels"]
            },
            "prediction": {
                "frame_indices": sample["frame_indices"],
                "phase_labels": pred_phase_ids
            },
            "raw_prediction": None
        }

        rows.append(out_row)

    with open(PRED_OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", PRED_OUT)


if __name__ == "__main__":

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Please run train_phase_head.py or train_phase_lora.py first."
        )
    run_inference()