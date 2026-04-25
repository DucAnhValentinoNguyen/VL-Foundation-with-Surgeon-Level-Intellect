import os
import json

test_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing"
phase_out = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/test_phase_task.jsonl"
anchor_out = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/test_anchor_task.jsonl"


PRIOR_RULE = (
    "Use the following prior information and label mappings. "
    "These mappings are provided only to explain the meaning of each integer ID. "
    "In the final JSON output, you must use integer IDs only. "
    "Do not output category names, text labels, or descriptions in the final JSON. "
)

PHASE_PRIOR = (
    "Phase ID mapping: "
    "0=Preparation; "
    "1=Calot triangle dissection; "
    "2=Clipping and cutting; "
    "3=Gallbladder dissection; "
    "4=Gallbladder retraction; "
    "5=Cleaning and coagulation; "
    "6=Gallbladder packaging. "
)

INSTRUMENT_PRIOR = (
    "Instrument ID mapping: "
    "0=grasper; "
    "1=bipolar; "
    "2=hook; "
    "3=scissors; "
    "4=clipper; "
    "5=irrigator; "
    "6=specimen-bag. "
)

OPERATOR_PRIOR = (
    "Operator ID mapping: "
    "0=null; "
    "1=main-surgeon-left-hand; "
    "2=assistant-surgeon-right-hand; "
    "3=main-surgeon-right-hand. "
)
JSON_ONLY_RULE = (
    "Do not output markdown. "
    "Do not output code fences. "
    "Do not output explanations. "
    "Output JSON only. "

)

def build_phase_segments(sample):

    annotated_frame_ids = sorted(sample["annotated_frame_ids_local"])
    annotations_in_clip = sample["annotations_in_clip"]
    num_frames_in_clip = sample["num_frames_in_clip"]

    if len(annotated_frame_ids) == 0:
        return {"phase_segments": []}

    frame_phase_pairs = []
    for fid in annotated_frame_ids:
        ann_list = annotations_in_clip[str(fid)]
        phase = ann_list[0]["phase"] if len(ann_list) > 0 else -1
        frame_phase_pairs.append((fid, phase))

    segments = []
    current_phase = frame_phase_pairs[0][1]
    seg_start = 0

    for i in range(1, len(frame_phase_pairs)):
        curr_fid, curr_phase = frame_phase_pairs[i]

        if curr_phase != current_phase:
            seg_end = curr_fid - 1
            segments.append({
                "start_frame": int(seg_start),
                "end_frame": int(seg_end),
                "phase": int(current_phase)
            })
            seg_start = curr_fid
            current_phase = curr_phase

    segments.append({
        "start_frame": int(seg_start),
        "end_frame": int(num_frames_in_clip - 1),
        "phase": int(current_phase)
    })

    return {"phase_segments": segments}

def round_bbox(bbox, ndigits=4):
    return [round(float(x), ndigits) for x in bbox]


def build_phase_prompt(num_frames_in_clip):
    return (
        "<video>\n"
        f"{PRIOR_RULE}"
        f"{PHASE_PRIOR}"
        f"The video has frames indexed from 0 to {num_frames_in_clip - 1}. "
        "Return only a valid JSON object with exactly one key: 'phase_segments'. "
        "The value of 'phase_segments' must be a list of segments that cover the whole video "
        "from frame 0 to the last frame. "
        "Each segment must contain exactly these keys: "
        "'start_frame', 'end_frame', 'phase'. "
        "The segments must be continuous, non-overlapping, and sorted by time. "
        "The 'phase' value must be an integer ID only, using the phase ID mapping above. "
        "Do not use phase names or words in the final JSON. "
        "Do not return a list directly. "
        f"{JSON_ONLY_RULE}"
        "Return a JSON object of the form: "
        "{\"phase_segments\": [{\"start_frame\": 0, \"end_frame\": 100, \"phase\": 1}]}"
    )


def build_anchor_frames(sample, seconds_stride=10):
    annotations_in_clip = sample["annotations_in_clip"]
    annotated_frame_ids = sorted(sample["annotated_frame_ids_local"])
    fps = float(sample["fps"])
    num_frames_in_clip = int(sample["num_frames_in_clip"])

    if len(annotated_frame_ids) == 0:
        return {"frames": []}

    # 每 10 秒取一个目标帧
    target_frame_ids = []
    sec = 0
    while True:
        target_fid = int(round(sec * seconds_stride * fps))
        if target_fid >= num_frames_in_clip:
            break
        target_frame_ids.append(target_fid)
        sec += 1

    frames = []
    for target_fid in target_frame_ids:
        nearest_fid = min(annotated_frame_ids, key=lambda x: abs(x - target_fid))
        ann_list = annotations_in_clip[str(nearest_fid)]
        phase = ann_list[0]["phase"] if len(ann_list) > 0 else -1

        objects = []
        for ann in ann_list:
            objects.append({
                "instrument": int(ann["instrument"]),
                "tool_bbox": round_bbox(ann["tool_bbox"]),
                "operator": int(ann["operator"]),
                "intraoperative_track": int(ann["intraoperative_track"]),
                "intracorporeal_track": int(ann["intracorporeal_track"]),
            })

        frames.append({
            "frame_index": int(target_fid),
            "phase": int(phase),
            "objects": objects
        })

    return {"frames": frames}


def build_anchor_prompt(frame_indices):
    return (
        "<video>\n"
        f"{PRIOR_RULE}"
        f"{PHASE_PRIOR}"
        f"{INSTRUMENT_PRIOR}"
        f"{OPERATOR_PRIOR}"
        "Return only one valid JSON object. "
        "The top-level output must be a JSON object, not a JSON array. "
        "The JSON object must contain exactly one key: 'frames'. "
        f"The value of 'frames' must be a list containing exactly one item for each of these frame indices sampled every 10 seconds: {frame_indices}. "
        "Do not omit any requested frame index. "
        "Do not invent any extra frame index. "
        "Each item in 'frames' must be a JSON object with exactly these keys: "
        "'frame_index', 'phase', 'objects'. "
        "The 'frame_index' value must be an integer and must equal one of the requested frame indices. "
        "The 'phase' value must be an integer ID only, using the phase ID mapping above. "
        "The 'objects' value must be a list. "
        "If no object is visible in a requested frame, return 'objects': []. "
        "Each item in 'objects' must be a JSON object with exactly these keys: "
        "'instrument', 'tool_bbox', 'operator', 'intraoperative_track', 'intracorporeal_track'. "
        "Every object must include all five keys. "
        "The 'instrument' value must be an integer ID only, using the instrument ID mapping above. "
        "The 'operator' value must be an integer ID only, using the operator ID mapping above. "
        "The 'intraoperative_track' value must be an integer ID only. "
        "The 'intracorporeal_track' value must be an integer ID only. "
        "The value of 'tool_bbox' must be a list of exactly 4 numbers in normalized [x, y, w, h] format. "
        "Do not use class names or text labels in the final JSON. "
        f"{JSON_ONLY_RULE}"
    )


files = sorted(os.listdir(test_dir))
mp4_files = [f for f in files if f.endswith(".mp4") and "_clip_" in f]

phase_rows = []
anchor_rows = []

for mp4_name in mp4_files:
    base = os.path.splitext(mp4_name)[0]
    json_name = base + ".json"

    mp4_path = os.path.join(test_dir, mp4_name)
    json_path = os.path.join(test_dir, json_name)

    if not os.path.exists(json_path):
        print("[WARNING] missing json for:", mp4_name)
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    if sample["num_annotated_frames"] == 0:
        print("[SKIP] no annotations:", json_name)
        continue

    # Prompt A
    phase_gt = build_phase_segments(sample)
    phase_rows.append({
        "video": mp4_path,
        "prompt": build_phase_prompt(sample["num_frames_in_clip"]),
        "ground_truth": phase_gt,
        "clip_json_path": json_path
    })

    # Prompt B
    anchor_gt = build_anchor_frames(sample, seconds_stride=10)
    frame_indices = [x["frame_index"] for x in anchor_gt["frames"]]

    anchor_rows.append({
        "video": mp4_path,
        "prompt": build_anchor_prompt(frame_indices),
        "ground_truth": anchor_gt,
        "clip_json_path": json_path
    })

with open(phase_out, "w", encoding="utf-8") as f:
    for row in phase_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(anchor_out, "w", encoding="utf-8") as f:
    for row in anchor_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("saved phase task:", phase_out)
print("num phase samples:", len(phase_rows))

print("saved anchor task:", anchor_out)
print("num anchor samples:", len(anchor_rows))

if len(phase_rows) > 0:
    print("\n===== Phase task example =====")
    print(json.dumps(phase_rows[0], indent=2, ensure_ascii=False))

if len(anchor_rows) > 0:
    print("\n===== Anchor task example =====")
    print(json.dumps(anchor_rows[0], indent=2, ensure_ascii=False))