import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from qwen_vl_utils import process_vision_info


class QwenVLLoraDataset(Dataset):
    """
    Dataset for Qwen2.5-VL LoRA fine-tuning.
    """

    def __init__(
        self,
        json_path: Path,
        project_root: Path,
        max_samples: Optional[int] = None,
    ):
        self.json_path = Path(json_path)
        self.project_root = Path(project_root)

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples is not None:
            data = data[:max_samples]

        self.data = data

    def __len__(self):
        return len(self.data)

    def _fix_image_path(self, image_path: str) -> str:
        filename = Path(image_path).name
        return str(self.project_root / "ds" / "img" / filename)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample["messages"]

        fixed_messages = []

        for msg in messages:
            new_msg = {"role": msg["role"], "content": []}

            for item in msg["content"]:
                if item["type"] == "image":
                    new_msg["content"].append({
                        "type": "image",
                        "image": self._fix_image_path(item["image"]),
                    })
                else:
                    new_msg["content"].append(item)

            fixed_messages.append(new_msg)

        return {
            "sample_id": sample["sample_id"],
            "messages": fixed_messages,
        }


class QwenVLCollator:
    """
    Collator for batch size 1.
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # This simple collator assumes per_device_train_batch_size=1.
        item = batch[0]
        messages = item["messages"]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()

        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

        return inputs


class QwenVLLoraTrainer:
    """
    Minimal QLoRA trainer for Qwen2.5-VL expert surgical communication.
    """

    def __init__(
        self,
        model_name: str,
        train_json_path: Path,
        val_json_path: Path,
        output_dir: Path,
        project_root: Path,
        max_train_samples: int = 50,
        max_val_samples: int = 20,
    ):
        self.model_name = model_name
        self.train_json_path = Path(train_json_path)
        self.val_json_path = Path(val_json_path)
        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

        self.model = None
        self.processor = None

    def load_model_and_processor(self):
        print(f"[INFO] Loading model for QLoRA: {self.model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print("[INFO] Model and processor ready.")

    def train(self):
        if self.model is None or self.processor is None:
            self.load_model_and_processor()

        train_dataset = QwenVLLoraDataset(
            json_path=self.train_json_path,
            project_root=self.project_root,
            max_samples=self.max_train_samples,
        )

        val_dataset = QwenVLLoraDataset(
            json_path=self.val_json_path,
            project_root=self.project_root,
            max_samples=self.max_val_samples,
        )

        collator = QwenVLCollator(
            processor=self.processor,
            max_length=2048,
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=20,
            save_steps=20,
            save_total_limit=2,
            fp16=True,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        print("[INFO] Starting LoRA fine-tuning...")
        trainer.train()

        final_adapter_dir = self.output_dir / "final_adapter"
        final_adapter_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(final_adapter_dir)
        self.processor.save_pretrained(final_adapter_dir)

        print(f"[INFO] Saved LoRA adapter to: {final_adapter_dir}")

        return final_adapter_dir