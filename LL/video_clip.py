import os
import json
import math
import cv2

# =========================
# 
# =========================
root_dir = "/content/drive/MyDrive/CholecTrack20/Testing" 
output_root = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips"
clip_minutes = 15
min_keep_minutes = 10  
# =========================

os.makedirs(output_root, exist_ok=True)

def find_one_file_by_ext(folder, exts):
    files = []
    for f in os.listdir(folder):
        lower = f.lower()
        if any(lower.endswith(ext) for ext in exts):
            files.append(os.path.join(folder, f))
    return sorted(files)

def save_video_clip(video_path, out_path, start_frame, end_frame, fps, width, height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame
    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    writer.release()
    cap.release()

def build_clip_json(data, clip_id, source_video_path, start_frame, end_frame, fps, width, height):
    annotations = data["annotations"]
    video_info = data.get("video", {})
    info = data.get("info", {})
    categories = data.get("categories", {})

    ann_keys = sorted([int(k) for k in annotations.keys()])


    ann_in_clip_original = {
        str(k): annotations[str(k)]
        for k in ann_keys
        if start_frame <= k <= end_frame
    }


    ann_in_clip_local = {}
    for k_str, ann_list in ann_in_clip_original.items():
        k = int(k_str)
        local_k = k - start_frame
        ann_in_clip_local[str(local_k)] = ann_list

    clip_json = {
        "clip_id": clip_id,
        "video_name": os.path.basename(source_video_path).rsplit(".", 1)[0],
        "source_video_path": source_video_path,

        "start_frame_original": int(start_frame),
        "end_frame_original": int(end_frame),

        "start_time_sec": float(start_frame / fps),
        "end_time_sec": float(end_frame / fps),

        "fps": float(fps),
        "width": int(width),
        "height": int(height),

        "num_frames_in_clip": int(end_frame - start_frame + 1),
        "num_annotated_frames": len(ann_in_clip_original),


        "annotated_frame_ids_original": [int(k) for k in ann_in_clip_original.keys()],
        "annotations_in_clip_original": ann_in_clip_original,

        "annotated_frame_ids_local": sorted([int(k) for k in ann_in_clip_local.keys()]),
        "annotations_in_clip_local": ann_in_clip_local,

        "info": info,
        "categories": categories,
        "source_video_meta": video_info,
    }
    return clip_json

subfolders = sorted([
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])

print("Found subfolders:", len(subfolders))

all_clip_count = 0

for folder in subfolders:
    folder_name = os.path.basename(folder)

    mp4_files = find_one_file_by_ext(folder, [".mp4"])
    json_files = find_one_file_by_ext(folder, [".json"])

    if len(mp4_files) == 0 or len(json_files) == 0:
        print(f"[SKIP] {folder_name}: missing mp4 or json")
        continue

    video_path = mp4_files[0]
    json_path = json_files[0]

    print("\n" + "=" * 80)
    print("Processing folder:", folder_name)
    print("video_path:", video_path)
    print("json_path:", json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or num_frames_video <= 0:
        print(f"[SKIP] Invalid video info for: {video_path}")
        continue

    clip_len_frames = int(clip_minutes * 60 * fps)
    min_keep_frames = int(min_keep_minutes * 60 * fps)

    full_clips = num_frames_video // clip_len_frames
    remainder = num_frames_video % clip_len_frames


    out_dir = os.path.join(output_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    clip_ranges = []


    for i in range(full_clips):
        start_frame = i * clip_len_frames
        end_frame = (i + 1) * clip_len_frames - 1
        clip_ranges.append((start_frame, end_frame))


    if remainder > min_keep_frames:
        start_frame = full_clips * clip_len_frames
        end_frame = num_frames_video - 1
        clip_ranges.append((start_frame, end_frame))

    print("fps:", fps)
    print("num_frames_video:", num_frames_video)
    print("clip_len_frames:", clip_len_frames)
    print("remainder_frames:", remainder)
    print("num_output_clips:", len(clip_ranges))

    for i, (start_frame, end_frame) in enumerate(clip_ranges, start=1):
        clip_id = f"{folder_name}_clip_{i:02d}"
        clip_mp4_path = os.path.join(out_dir, clip_id + ".mp4")
        clip_json_path = os.path.join(out_dir, clip_id + ".json")

        save_video_clip(
            video_path=video_path,
            out_path=clip_mp4_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            width=width,
            height=height,
        )

        clip_json = build_clip_json(
            data=data,
            clip_id=clip_id,
            source_video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            width=width,
            height=height,
        )

        with open(clip_json_path, "w", encoding="utf-8") as f:
            json.dump(clip_json, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] {clip_mp4_path}")
        print(f"[SAVED] {clip_json_path}")
        all_clip_count += 1

print("\n" + "=" * 80)
print("All done.")
print("Total clips created:", all_clip_count)
print("Output root:", output_root)