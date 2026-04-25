import os
import json


data_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing"
out_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/finetune_long_video_10s.jsonl"
sample_stride_sec = 10



def round_bbox(bbox, ndigits=4):
    return [round(float(x), ndigits) for x in bbox]


def get_target_frame_indices(num_frames_in_clip, fps, stride_sec=10):
    frame_indices = []
    sec = 0
    while True:
        fid = int(round(sec * stride_sec * fps))
        if fid >= num_frames_in_clip:
            break
        frame_indices.append(fid)
        sec += 1
    return frame_indices


def build_one_video_training_row(mp4_path, json_path, sample, stride_sec=10):
    annotated_frame_ids = sorted(sample["annotated_frame_ids_local"])
    annotations_in_clip = sample["annotations_in_clip"]
    fps = float(sample["fps"])
    num_frames_in_clip = int(sample["num_frames_in_clip"])

    if len(annotated_frame_ids) == 0:
        return None

    target_frame_indices = get_target_frame_indices(
        num_frames_in_clip=num_frames_in_clip,
        fps=fps,
        stride_sec=stride_sec
    )

    phase_labels = []
    anchor_targets = []

    for target_fid in target_frame_indices:
        nearest_fid = min(annotated_frame_ids, key=lambda x: abs(x - target_fid))
        ann_list = annotations_in_clip[str(nearest_fid)]

        phase = ann_list[0]["phase"] if len(ann_list) > 0 else -1
        phase_labels.append(int(phase))

        objects = []
        for ann in ann_list:
            objects.append({
                "instrument": int(ann["instrument"]),
                "tool_bbox": round_bbox(ann["tool_bbox"]),
                "operator": int(ann["operator"]),
                "intraoperative_track": int(ann["intraoperative_track"]),
                "intracorporeal_track": int(ann["intracorporeal_track"]),
            })

        anchor_targets.append({
            "frame_index": int(target_fid),
            "phase": int(phase),
            "objects": objects
        })

    row = {
        "video": mp4_path,
        "clip_json_path": json_path,
        "fps": fps,
        "num_frames_in_clip": num_frames_in_clip,
        "sample_stride_sec": int(stride_sec),
        "frame_indices": [int(x) for x in target_frame_indices], 
        "phase_labels": phase_labels,                  
        "anchor_targets": anchor_targets                           
    }
    return row


files = sorted(os.listdir(data_dir))
mp4_files = [f for f in files if f.endswith(".mp4") and "_clip_" in f]

rows = []

for mp4_name in mp4_files:
    base = os.path.splitext(mp4_name)[0]
    json_name = base + ".json"

    mp4_path = os.path.join(data_dir, mp4_name)
    json_path = os.path.join(data_dir, json_name)

    if not os.path.exists(json_path):
        print("[WARNING] missing json for:", mp4_name)
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    if sample["num_annotated_frames"] == 0:
        print("[SKIP] no annotations:", json_name)
        continue

    row = build_one_video_training_row(
        mp4_path=mp4_path,
        json_path=json_path,
        sample=sample,
        stride_sec=sample_stride_sec
    )

    if row is not None:
        rows.append(row)

with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("saved:", out_path)
print("num videos:", len(rows))

if len(rows) > 0:
    print("\n===== Example =====")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))