import json
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from utils import safe_parse_json, schema_valid_phase_segments, segments_to_sampled_phase_labels
from build_gt_jsonl import process_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRED_OUT = DATA_DIR / "results" / "zeroshot_predictions.jsonl"
PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

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

JSON_ONLY_RULE = (
    "Do not output markdown. "
    "Do not output code fences. "
    "Do not output explanations. "
    "Output JSON only. "
)

def build_phase_prompt(num_frames_in_clip: int) -> str:
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



def predict_zeroshot(sample, model, processor):
    video_path = sample["video"]
    num_frames_in_clip = int(sample["num_frames_in_clip"])
    prompt = build_phase_prompt(num_frames_in_clip).replace("<video>\n", "").replace("<video>", "").strip()

    if not video_path.startswith("file://"):
        video_path_for_msg = "file://" + video_path
    else:
        video_path_for_msg = video_path

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path_for_msg,
                    "fps": 0.25,
                    "min_pixels": 256 * 256,
                    "max_pixels": 512 * 512,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True
    )

    fixed_video_kwargs = {}
    for k, v in video_kwargs.items():
        if isinstance(v, list) and len(v) == 1:
            fixed_video_kwargs[k] = v[0]
        else:
            fixed_video_kwargs[k] = v

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **fixed_video_kwargs
    )

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    input_token_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_token_len:]
    raw_pred = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    parsed_pred = safe_parse_json(raw_pred)

    return raw_pred, parsed_pred, prompt

def run_zeroshot_phase(test_jsonl, model_name, pred_out):

    dataset = load_dataset("json", data_files=test_jsonl)["train"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    rows = []

    for i in range(len(dataset)):
        print(f"running zero-shot phase task {i+1}/{len(dataset)}")
        sample = dataset[i]
        raw_pred, parsed_pred, prompt = predict_zeroshot(sample, model, processor)

        gt_frame_indices = sample["frame_indices"]
        gt_phase_labels = sample["phase_labels"]

        pred_structured = None
        if schema_valid_phase_segments(parsed_pred):
            pred_phase_labels = segments_to_sampled_phase_labels(
                parsed_pred["phase_segments"],
                gt_frame_indices
            )
            pred_structured = {
                "frame_indices": gt_frame_indices,
                "phase_labels": pred_phase_labels
            }

        out_row = {
            "video": sample["video"],
            "clip_json_path": sample.get("clip_json_path", None),
            "prompt": prompt,
            "ground_truth": {
                "frame_indices": gt_frame_indices,
                "phase_labels": gt_phase_labels
            },
            "prediction": pred_structured,
            "raw_prediction": raw_pred
        }

        rows.append(out_row)

    with open(pred_out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", pred_out)


if __name__ == "__main__":
    

    test_dir = DATA_DIR / "testing"
    gt_name = "gt_10s.jsonl"
    test_jsonl = test_dir / gt_name

    process_split(test_dir, gt_name=gt_name, sample_stride_sec=10)
    run_zeroshot_phase(str(test_jsonl), MODEL_NAME, str(PRED_OUT))