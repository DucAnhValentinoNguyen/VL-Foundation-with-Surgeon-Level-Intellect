import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenZeroShotRunner:
    """
    Runs Qwen2.5-VL zero-shot inference for Task 5.
    """

    def __init__(
        self,
        model_name: str,
        input_json_path: Path,
        output_json_path: Path,
        project_root: Optional[Path] = None,
        max_samples: int = 5,
    ):
        self.model_name = model_name
        self.input_json_path = Path(input_json_path)
        self.output_json_path = Path(output_json_path)
        self.project_root = Path(project_root) if project_root is not None else None
        self.max_samples = max_samples

        self.model = None
        self.processor = None

    def load_samples(self) -> List[Dict[str, Any]]:
        if not self.input_json_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {self.input_json_path}")

        with open(self.input_json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        return samples[: self.max_samples]

    def fix_image_path(self, sample_or_path) -> str:
        if self.project_root is None:
            if isinstance(sample_or_path, dict):
                return sample_or_path["image_path"]
            return str(sample_or_path)

        if isinstance(sample_or_path, dict):
            filename = sample_or_path.get("image_filename")
            if filename is not None:
                return str(self.project_root / "ds" / "img" / filename)

            old_path = sample_or_path["image_path"]
            filename = Path(old_path).name
            return str(self.project_root / "ds" / "img" / filename)

        filename = Path(sample_or_path).name
        return str(self.project_root / "ds" / "img" / filename)

    def load_model(self):
        print(f"[INFO] Loading model: {self.model_name}")

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        print("[INFO] Model loaded successfully.")

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
            "model_name": self.model_name,
        }

    def run(self) -> List[Dict[str, Any]]:
        samples = self.load_samples()

        if self.model is None or self.processor is None:
            self.load_model()

        predictions = []

        for idx, sample in enumerate(samples, start=1):
            print(f"[INFO] Running zero-shot {idx}/{len(samples)}: {sample['sample_id']}")

            try:
                pred = self.run_single(sample)
                predictions.append(pred)
            except Exception as e:
                print(f"[ERROR] Failed sample {sample.get('sample_id')}: {e}")

        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved predictions to: {self.output_json_path}")

        return predictions