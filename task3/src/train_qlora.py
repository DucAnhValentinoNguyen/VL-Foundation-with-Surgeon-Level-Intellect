import torch
import json
import os
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv

# Load HF_TOKEN from .env
load_dotenv()

# --- 1. Custom Data Collator for Qwen2.5-VL ---
class Qwen25VLDataCollator:
    """Dynamically processes images and text into Qwen's specific tensor format during training."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        image_inputs_list = []
        # video_inputs_list = []
        
        for example in examples:
            # Reconstruct the multimodal messages array
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{os.path.abspath(example['image'])}"},
                        {"type": "text", "text": example['user_prompt']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example['assistant_response']}
                    ]
                }
            ]
            
            # Apply Qwen's specific chat template for training
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            
            texts.append(text)
            image_inputs_list.append(image_inputs)
            # video_inputs_list.append(video_inputs)
            
        # The processor handles the heavy lifting of pixel values and tokenization
        batch = self.processor(
            text=texts,
            images=image_inputs_list,
            # videos=video_inputs_list,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels for causal language modeling
        labels = batch["input_ids"].clone()
        # Mask the padding tokens so the model doesn't train on them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

def train_surg_intellect():
    print("Initializing QLoRA Training for Qwen2.5-VL-3B-Instruct on RTX 4090...")
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 2. 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 3. Load Model in 4-bit
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Prepare model for memory-efficient training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Inject LoRA Adapters
    # We target the specific Attention (q_proj, v_proj) and MLP (gate_proj, up_proj, down_proj) layers
    # This teaches the model to link clinical text (tools) to visual features
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / all_param:.2f}% of model)")

    # 5. Load Training Data (From dataset.py)
    with open("../data/surgvu/surg_vlm_train.jsonl", "r") as f:
        # We load a subset (e.g., 500 samples) so it finishes quickly for your weekend demo
        # If you have time, let it run on all 4,283 samples!
        raw_data = [json.loads(line) for line in f.readlines()[:500]]
        
    formatted_data = []
    for d in raw_data:
        formatted_data.append({
            "image": d['image'],
            "user_prompt": d['conversations'][0]['value'],
            "assistant_response": d['conversations'][1]['value']
        })
        
    dataset = Dataset.from_list(formatted_data)
    data_collator = Qwen25VLDataCollator(processor)

    # 6. Training Arguments optimized for 24GB VRAM
    training_args = TrainingArguments(
        output_dir="./surg_intellect_checkpoints",
        per_device_train_batch_size=2,   # Batch size of 2 fits well on a 4090 for 3B
        gradient_accumulation_steps=4,   # Effective batch size = 8
        learning_rate=2e-4,
        warmup_ratio=0.1,
        max_steps=100,                   # Fast run for the demo. Increase to 500+ for actual results.
        logging_steps=10,
        save_strategy="no",              # Don't clutter drive during demo
        bf16=True,                       # Hardware native precision for 4090
        optim="paged_adamw_32bit",
        remove_unused_columns=False,     # CRITICAL for multimodal datasets
        report_to="none"                 # Disable wandb/tensorboard for simplicity
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nStarting the Surgical Adaptation...")
    trainer.train()
    
    # 8. Save the surgical brain!
    output_path = "./surg_intellect_final_adapter"
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"\nSuccess! Surgeon-level adapters saved to {output_path}")

if __name__ == "__main__":
    train_surg_intellect()