import torch
import json
import os
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv

# Load HF_TOKEN from .env
load_dotenv()

def run_zero_shot_baseline(test_manifest_path, output_path):
    print("Initializing Qwen2.5-VL-3B-Instruct in 4-bit quantization on RTX 4090...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Use AutoProcessor and the specific Vision class
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # CRITICAL: Using Qwen2_5_VLForConditionalGeneration instead of AutoModelForCausalLM
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    with open(test_manifest_path, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()][::5]
        
    print(f"\nModel loaded successfully! Starting evaluation on {len(samples)} test samples...")
    print("-" * 60)

    with open(output_path, 'w') as out_f:
        # Define a batch size
        batch_size = 4 
        # Process in chunks
        for i in tqdm(range(0, len(samples), batch_size), desc="Zero-Shot Batch Inference"):
            batch_samples = samples[i : i + batch_size]
            
            batch_messages = []
            for sample in batch_samples:
                batch_messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{os.path.abspath(sample['image'])}"},
                            {"type": "text", "text": sample['conversations'][0]['value']}
                        ]
                    }
                ])

            # Prep inputs for the whole batch
            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                # Add these two lines to cap the resolution
                min_pixels=256 * 28 * 28, 
                max_pixels=512 * 28 * 28, 
            ).to("cuda")


            # Batch Inference
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                
            # Decode and save
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for sample, response in zip(batch_samples, responses):
                result_entry = {
                    "id": sample.get("id", "unknown"),
                    "prediction": response.strip(),
                    "ground_truth": sample['conversations'][1]['value']
                }
                out_f.write(json.dumps(result_entry) + "\n")
                out_f.flush()

    print(f"\nBaseline evaluation complete! Results saved to {output_path}")

if __name__ == "__main__":
    TEST_MANIFEST = "../data/surgvu/surg_vlm_test.jsonl"
    OUTPUT_FILE = "zero_shot_results.jsonl"
    
    if os.path.exists(TEST_MANIFEST):
        run_zero_shot_baseline(TEST_MANIFEST, OUTPUT_FILE)
    else:
        print(f"Could not find test manifest at {TEST_MANIFEST}")