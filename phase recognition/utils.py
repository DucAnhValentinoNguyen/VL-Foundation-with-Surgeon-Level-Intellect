# Utilities for zero-shot phase prediction post-processing:
# 1) safely parse raw model output as JSON,
# 2) validate whether the parsed JSON follows the expected phase-segment schema,
# 3) convert predicted phase segments into sampled frame-wise phase labels
#    so they can be aligned with ground truth for evaluation.

import json

def safe_parse_json(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    x = x.strip()
    try:
        return json.loads(x)
    except Exception:
        return None


def schema_valid_phase_segments(obj):
    if not isinstance(obj, dict):
        return False
    segs = obj.get("phase_segments")
    if not isinstance(segs, list):
        return False
    for seg in segs:
        if not isinstance(seg, dict):
            return False
        required = {"start_frame", "end_frame", "phase"}
        if not required.issubset(seg.keys()):
            return False
    return True


def segments_to_sampled_phase_labels(segments, frame_indices):
    pred_labels = []
    for fid in frame_indices:
        assigned = -1
        for seg in segments:
            s = int(seg["start_frame"])
            e = int(seg["end_frame"])
            p = int(seg["phase"])
            if s <= fid <= e:
                assigned = p
                break
        pred_labels.append(assigned)
    return pred_labels