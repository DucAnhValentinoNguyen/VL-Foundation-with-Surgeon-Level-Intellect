import torch
import json
import os
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

def run_adapter_inference(test_manifest_path, adapter_path, output_path):
    print(f"Loading Fine-Tuned SurgIntellect Model from {adapter_path}...")
    
    # 1. Load Base Model in 4-bit
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 2. Load and Merge LoRA Adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    with open(test_manifest_path, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()][::5] # Subsample for speed

    print(f"Running Inference on {len(samples)} test samples...")

    with open(output_path, 'w') as out_f:
        batch_size = 4
        for i in tqdm(range(0, len(samples), batch_size)):
            batch_samples = samples[i : i + batch_size]
            
            batch_messages = []
            for s in batch_samples:
                batch_messages.append([
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{os.path.abspath(s['image'])}"},
                        {"type": "text", "text": s['conversations'][0]['value']}
                    ]}
                ])

            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_inputs, _ = process_vision_info(batch_messages)
            
            inputs = processor(
                text=texts, images=image_inputs, padding=True, return_tensors="pt",
                min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
            
            trimmed_ids = [out[len(in_):] for in_, out in zip(inputs.input_ids, generated_ids)]
            responses = processor.batch_decode(trimmed_ids, skip_special_tokens=True)

            for sample, response in zip(batch_samples, responses):
                out_f.write(json.dumps({
                    "id": sample['id'],
                    "prediction": response.strip(),
                    "ground_truth": sample['conversations'][1]['value']
                }) + "\n")

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_adapter_inference("../data/surgvu/surg_vlm_test.jsonl", "./surg_intellect_final_adapter", "finetuned_results.jsonl")