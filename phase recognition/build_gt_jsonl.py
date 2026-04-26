import os
import json


def get_target_frame_indices(num_frames_in_clip, fps, stride_sec=10):
    # Sample one target frame every `stride_sec` seconds.
    frame_indices = []
    sec = 0
    while True:
        fid = int(round(sec * stride_sec * fps))
        if fid >= num_frames_in_clip:
            break
        frame_indices.append(fid)
        sec += 1
    return frame_indices


def build_one_video_phase_row(mp4_path, json_path, sample, stride_sec=10):
    # Build one JSONL row containing sampled frame indices and phase labels for a single video.
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

    for target_fid in target_frame_indices:
        nearest_fid = min(annotated_frame_ids, key=lambda x: abs(x - target_fid))
        ann_list = annotations_in_clip[str(nearest_fid)]

        phase = ann_list[0]["phase"] if len(ann_list) > 0 else -1
        phase_labels.append(int(phase))

    row = {
        "video": mp4_path,
        "clip_json_path": json_path,
        "fps": fps,
        "num_frames_in_clip": num_frames_in_clip,
        "sample_stride_sec": int(stride_sec),
        "frame_indices": [int(x) for x in target_frame_indices],
        "phase_labels": phase_labels
    }
    return row


def process_split(data_dir, gt_name="gt_10s.jsonl", sample_stride_sec=10):
    out_path = os.path.join(data_dir, gt_name)

    if not os.path.exists(data_dir):
        print(f"[WARNING] split dir not found: {data_dir}")
        return

    files = sorted(os.listdir(data_dir))
    mp4_files = [f for f in files if f.endswith(".mp4") and "_clip_" in f]

    rows = []

    for mp4_name in mp4_files:
        base = os.path.splitext(mp4_name)[0]
        json_name = base + ".json"

        mp4_path = os.path.join(data_dir, mp4_name)
        json_path = os.path.join(data_dir, json_name)

        if not os.path.exists(json_path):
            print(f"[WARNING] missing json for: {mp4_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        if sample["num_annotated_frames"] == 0:
            print(f"[SKIP] no annotations: {json_path}")
            continue

        row = build_one_video_phase_row(
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

    print(f"saved: {out_path}")
    print(f"num videos in {data_dir}: {len(rows)}")


