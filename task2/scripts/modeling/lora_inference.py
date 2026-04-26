import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info


class QwenLoraInferenceRunner:
    """
    Runs inference with a Qwen2.5-VL base model + trained LoRA adapter.
    """

    def __init__(
        self,
        model_name: str,
        adapter_path: Path,
        input_json_path: Path,
        output_json_path: Path,
        project_root: Path,
        max_samples: int = 5,
    ):
        self.model_name = model_name
        self.adapter_path = Path(adapter_path)
        self.input_json_path = Path(input_json_path)
        self.output_json_path = Path(output_json_path)
        self.project_root = Path(project_root)
        self.max_samples = max_samples

        self.model = None
        self.processor = None

    def load_samples(self) -> List[Dict[str, Any]]:
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples[: self.max_samples]

    def fix_image_path(self, sample: Dict[str, Any]) -> str:
        filename = sample.get("image_filename")

        if filename is None:
            filename = Path(sample["image_path"]).name

        return str(self.project_root / "ds" / "img" / filename)

    def load_model(self):
        print(f"[INFO] Loading base model: {self.model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        print(f"[INFO] Loading LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model.eval()
        print("[INFO] LoRA model ready.")

    @staticmethod
    def build_prompt() -> str:
        return """
You are an expert surgical vision-language assistant.

Analyze the laparoscopic cholecystectomy image.

Return your answer in valid JSON with these fields:
{
  "visible_instruments": [],
  "visible_anatomy_or_tissue": [],
  "visible_action": "",
  "possible_surgical_phase": "",
  "expert_surgical_description": "",
  "uncertainty_note": ""
}

Rules:
- Use only visible evidence from the image.
- Do not hallucinate tools, anatomy, bleeding, complications, or surgical phase.
- If the phase cannot be confirmed from a single frame, say "uncertain from this single frame".
- If anatomy is unclear, say "not clearly identifiable".
- Do not claim Critical View of Safety, clipping, cutting, or duct division unless clearly visible.
- Be cautious and clinically grounded.
"""

    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.fix_image_path(sample)
        prompt = self.build_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "sample_id": sample["sample_id"],
            "video_id": sample["video_id"],
            "sequence": sample["sequence"],
            "image_filename": sample.get("image_filename"),
            "image_path": image_path,
            "visible_classes": sample["visible_classes"],
            "visible_instruments_gt": sample["visible_instruments"],
            "visible_anatomy_or_tissue_gt": sample["visible_anatomy_or_tissue"],
            "teacher_answer": sample["teacher_answer"],
            "qwen_zero_shot_output": output_text,
            "model_name": f"{self.model_name}+LoRA",
        }

    def run(self) -> List[Dict[str, Any]]:
        samples = self.load_samples()

        if self.model is None or self.processor is None:
            self.load_model()

        predictions = []

        for idx, sample in enumerate(samples, start=1):
            print(f"[INFO] Running LoRA inference {idx}/{len(samples)}: {sample['sample_id']}")

            try:
                pred = self.run_single(sample)
                predictions.append(pred)
            except Exception as e:
                print(f"[ERROR] Failed sample {sample.get('sample_id')}: {e}")

        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved LoRA predictions to: {self.output_json_path}")

        return predictions