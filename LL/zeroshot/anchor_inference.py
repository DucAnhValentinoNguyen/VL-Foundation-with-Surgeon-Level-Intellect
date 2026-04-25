import json
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


test_pairs_path = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/test_anchor_task.jsonl"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
pred_out = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/testing/zeroshot_anchor_predictions.jsonl"


dataset = load_dataset("json", data_files=test_pairs_path)["train"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

def predict_one(sample):
    prompt = sample["prompt"].strip()
    video_path = sample["video"]

    prompt = prompt.replace("<video>\n", "").replace("<video>", "").strip()

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
                    "max_pixels": 512 * 512
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
            max_new_tokens=15000,
            do_sample=False
        )

    input_token_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_token_len:]
    pred = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return pred

rows = []
for i in range(len(dataset)):
    print(f"running anchor task {i+1}/{len(dataset)}")
    sample = dataset[i]
    pred = predict_one(sample)

    rows.append({
        "video": sample["video"],
        "prompt": sample["prompt"],
        "ground_truth": sample["ground_truth"],
        "prediction": pred,
        "clip_json_path": sample["clip_json_path"]
    })

with open(pred_out, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("saved:", pred_out)