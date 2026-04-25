import os
import json
import time
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLModel
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from LL.finetune.anchor_training import load_jsonl,read_frames_rgb,collate_fn, TaskBQwen25VLLoRA


test_jsonl = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetune_long_video_10s.jsonl"
ckpt_path = "/content/drive/MyDrive/CholecTrack20/taskB_qwen25vl_lora/last.pt"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
pred_out = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetuned_anchor_predictions.jsonl"

num_phase_classes = 7
num_instrument_classes = 7
num_operator_classes = 4
obj_threshold = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

max_frames_per_video = 90
lora_r = 4
lora_alpha = 8
lora_dropout = 0.05


# -------------------------
# Utils
# -------------------------

def round_bbox(bbox, ndigits=4):
    return [round(float(x), ndigits) for x in bbox]


# -------------------------
# Dataset
# -------------------------
class TaskBTestDataset(Dataset):
    def __init__(self, jsonl_path, max_frames_per_video=None):
        self.rows = load_jsonl(jsonl_path)
        self.max_frames_per_video = max_frames_per_video

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        video_path = row["video"]
        frame_indices = row["frame_indices"]
        anchor_targets = row["anchor_targets"]

        if self.max_frames_per_video is not None:
            frame_indices = frame_indices[:self.max_frames_per_video]
            anchor_targets = anchor_targets[:self.max_frames_per_video]

        frames = read_frames_rgb(video_path, frame_indices)

        return {
            "video": video_path,
            "frames": frames,
            "frame_indices": frame_indices,
            "anchor_targets": anchor_targets,
            "clip_json_path": row.get("clip_json_path", None),
        }




# -------------------------
# Decode prediction
# -------------------------
def decode_taskb_outputs(outputs, frame_indices, obj_threshold=0.5):
    phase_logits = outputs["phase_logits"][0]
    obj_logits = outputs["obj_logits"][0]
    instr_logits = outputs["instr_logits"][0]
    op_logits = outputs["op_logits"][0]
    bbox_pred = outputs["bbox_pred"][0]
    intraop_logits = outputs["intraop_logits"][0]
    intracorp_logits = outputs["intracorp_logits"][0]

    phase_pred = phase_logits.argmax(dim=-1).detach().cpu().tolist()
    obj_prob = torch.sigmoid(obj_logits).detach().cpu()
    instr_pred = instr_logits.argmax(dim=-1).detach().cpu().tolist()
    op_pred = op_logits.argmax(dim=-1).detach().cpu().tolist()
    intraop_pred = intraop_logits.argmax(dim=-1).detach().cpu().tolist()
    intracorp_pred = intracorp_logits.argmax(dim=-1).detach().cpu().tolist()
    bbox_pred = bbox_pred.detach().cpu().tolist()

    frames = []

    for t, fid in enumerate(frame_indices):
        objects = []
        k = len(obj_prob[t])

        for s in range(k):
            if float(obj_prob[t][s].item()) >= obj_threshold:
                objects.append({
                    "instrument": int(instr_pred[t][s]),
                    "tool_bbox": round_bbox(bbox_pred[t][s]),
                    "operator": int(op_pred[t][s]),
                    "intraoperative_track": int(intraop_pred[t][s]),
                    "intracorporeal_track": int(intracorp_pred[t][s]),
                })

        frames.append({
            "frame_index": int(fid),
            "phase": int(phase_pred[t]),
            "objects": objects
        })

    return {"frames": frames}


@torch.no_grad()
def run_inference():
    dataset = TaskBTestDataset(test_jsonl, max_frames_per_video=max_frames_per_video)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpt = torch.load(ckpt_path, map_location=device)

    model = TaskBQwen25VLLoRA(
        model_name=model_name,
        num_phase_classes=num_phase_classes,
        num_instrument_classes=num_instrument_classes,
        num_operator_classes=num_operator_classes,
        num_intraop_classes=ckpt["num_intraop_classes"],
        num_intracorp_classes=ckpt["num_intracorp_classes"],
        num_slots=ckpt["num_slots"],
        lora_r=ckpt.get("lora_r", lora_r),
        lora_alpha=ckpt.get("lora_alpha", lora_alpha),
        lora_dropout=ckpt.get("lora_dropout", lora_dropout),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    rows = []

    for sample in tqdm(loader, desc="Task B inference"):
        outputs = model(sample["frames"])
        prediction = decode_taskb_outputs(
            outputs=outputs,
            frame_indices=sample["frame_indices"],
            obj_threshold=obj_threshold
        )

        ground_truth = {"frames": sample["anchor_targets"]}

        rows.append({
            "video": sample["video"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "clip_json_path": sample["clip_json_path"]
        })

    with open(pred_out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", pred_out)


if __name__ == "__main__":
    run_inference()