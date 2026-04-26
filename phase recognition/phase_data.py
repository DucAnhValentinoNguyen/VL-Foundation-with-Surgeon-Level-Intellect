# Data utilities for phase recognition:
# - read JSONL metadata,
# - check video accessibility,
# - load sampled RGB frames from videos,
# - sample frame indices at a fixed temporal stride (optionally with random offset),
# - assign phase labels using the nearest annotated frame,
# - wrap everything into a Dataset for training and inference.

import os
import json
import cv2
import time
import random
import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def can_open_video(video_path):
    if not os.path.exists(video_path):
        return False
    cap = cv2.VideoCapture(video_path)
    ok = cap.isOpened()
    cap.release()
    return ok


def read_frames_rgb(video_path, frame_indices, max_retries=3, sleep_sec=1.0):
    last_error = None

    for attempt in range(max_retries):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            last_error = RuntimeError(f"Failed to open video: {video_path} | attempt={attempt+1}")
            cap.release()
            time.sleep(sleep_sec)
            continue

        frames = []
        ok_all = True

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                ok_all = False
                last_error = RuntimeError(
                    f"Failed to read frame {frame_index} from {video_path} | attempt={attempt+1}"
                )
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if ok_all:
            return frames

        time.sleep(sleep_sec)

    raise last_error


def get_target_frame_indices_with_offset(num_frames_in_clip, fps, stride_sec=10, offset_sec=0.0):
    frame_indices = []
    t = float(offset_sec)
    while True:
        fid = int(round(t * fps))
        if fid >= num_frames_in_clip:
            break
        frame_indices.append(fid)
        t += stride_sec
    return frame_indices


def build_phase_labels_from_sample(sample, target_frame_indices):
    annotated_frame_ids = sorted(sample["annotated_frame_ids_local"])
    annotations_in_clip = sample["annotations_in_clip"]

    phase_labels = []
    for target_fid in target_frame_indices:
        nearest_fid = min(annotated_frame_ids, key=lambda x: abs(x - target_fid))
        ann_list = annotations_in_clip[str(nearest_fid)]
        phase = ann_list[0]["phase"] if len(ann_list) > 0 else -1
        phase_labels.append(int(phase))
    return phase_labels


class TaskAPhaseDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        max_frames_per_video=None,
        num_phase_classes=7,
        validate=False,
        sample_stride_sec=10,
        random_offset_sec=True,
    ):
        raw_rows = load_jsonl(jsonl_path)
        self.rows = []
        self.max_frames_per_video = max_frames_per_video
        self.num_phase_classes = num_phase_classes
        self.validate = validate
        self.sample_stride_sec = sample_stride_sec
        self.random_offset_sec = random_offset_sec

        for row in raw_rows:
            video_path = row["video"]
            clip_json_path = row.get("clip_json_path", None)

            if validate:
                if not can_open_video(video_path):
                    print("[SKIP] bad video:", video_path)
                    continue

                if clip_json_path is None or not os.path.exists(clip_json_path):
                    print("[SKIP] missing clip_json_path:", video_path)
                    continue

                with open(clip_json_path, "r", encoding="utf-8") as f:
                    sample = json.load(f)

                annotated_frame_ids = sorted(sample["annotated_frame_ids_local"])
                if len(annotated_frame_ids) == 0:
                    print("[SKIP] no annotations:", video_path)
                    continue

            self.rows.append(row)

        print("num usable rows:", len(self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        video_path = row["video"]
        clip_json_path = row["clip_json_path"]

        with open(clip_json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        fps = float(sample["fps"])
        num_frames_in_clip = int(sample["num_frames_in_clip"])

        if self.random_offset_sec:
            offset_sec = random.random() * self.sample_stride_sec
        else:
            offset_sec = 0.0

        frame_indices = get_target_frame_indices_with_offset(
            num_frames_in_clip=num_frames_in_clip,
            fps=fps,
            stride_sec=self.sample_stride_sec,
            offset_sec=offset_sec,
        )

        if len(frame_indices) == 0:
            raise ValueError(f"No sampled frames for video: {video_path}")

        phase_labels = build_phase_labels_from_sample(sample, frame_indices)

        if self.max_frames_per_video is not None:
            frame_indices = frame_indices[:self.max_frames_per_video]
            phase_labels = phase_labels[:self.max_frames_per_video]

        frames = read_frames_rgb(video_path, frame_indices)

        return {
            "video": video_path,
            "video_path": video_path,
            "frames": frames,
            "frame_indices": frame_indices,
            "phase_labels": phase_labels,
            "clip_json_path": clip_json_path,
            "num_frames_in_clip": num_frames_in_clip,
            "offset_sec": offset_sec,
        }


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]