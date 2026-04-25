import os
import json
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLModel
from peft import LoraConfig, get_peft_model
from phase_training import TaskAQwen25VLLoRA,load_jsonl, read_frames_rgb, collate_fn, sampled_phases_to_segments


test_jsonl = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetune_long_video_10s.jsonl"

ckpt_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/taskA_qwen25vl_lora/last.pt"

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
pred_out = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetuned_phase_predictions.jsonl"

num_phase_classes = 7
device = "cuda" if torch.cuda.is_available() else "cpu"

max_frames_per_video = 90
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05





# def read_frames_rgb(video_path, frame_indices, max_retries=3, sleep_sec=1.0):
#     last_error = None

#     for attempt in range(max_retries):
#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             last_error = RuntimeError(f"Failed to open video: {video_path} | attempt={attempt+1}")
#             cap.release()
#             time.sleep(sleep_sec)
#             continue

#         frames = []
#         ok_all = True

#         for frame_index in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
#             ok, frame = cap.read()
#             if not ok or frame is None:
#                 ok_all = False
#                 last_error = RuntimeError(
#                     f"Failed to read frame {frame_index} from {video_path} | attempt={attempt+1}"
#                 )
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)

#         cap.release()

#         if ok_all:
#             return frames

#         time.sleep(sleep_sec)

#     raise last_error


class TaskATestDataset(Dataset):
    def __init__(self, jsonl_path, max_frames_per_video=None):
        self.rows = load_jsonl(jsonl_path)
        self.max_frames_per_video = max_frames_per_video

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        video_path = row["video"]
        frame_indices = row["frame_indices"]
        phase_labels = row["phase_labels"]

        if self.max_frames_per_video is not None:
            frame_indices = frame_indices[:self.max_frames_per_video]
            phase_labels = phase_labels[:self.max_frames_per_video]

        frames = read_frames_rgb(video_path, frame_indices)

        return {
            "video": video_path,
            "frames": frames,
            "frame_indices": frame_indices,
            "phase_labels": phase_labels,
            "clip_json_path": row.get("clip_json_path", None),
            "num_frames_in_clip": row.get("num_frames_in_clip", None),
        }



@torch.no_grad()
def run_inference():
    dataset = TaskATestDataset(test_jsonl, max_frames_per_video=max_frames_per_video)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TaskAQwen25VLLoRA(
        model_name=model_name,
        num_phase_classes=num_phase_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    rows = []

    for i, sample in enumerate(loader):
        print(f"running phase inference {i+1}/{len(dataset)}")

        phase_logits = model(sample["frames"])
        pred_phase_ids = phase_logits.argmax(dim=-1)[0].detach().cpu().tolist()

        prediction = sampled_phases_to_segments(
            frame_indices=sample["frame_indices"],
            pred_phase_ids=pred_phase_ids,
            num_frames_in_clip=sample["num_frames_in_clip"]
        )

        ground_truth = sampled_phases_to_segments(
            frame_indices=sample["frame_indices"],
            pred_phase_ids=sample["phase_labels"],
            num_frames_in_clip=sample["num_frames_in_clip"]
        )

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