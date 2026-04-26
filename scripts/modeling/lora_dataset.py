import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class LoraDatasetBuilder:
    """
    Converts teacher-labelled surgical communication samples into
    Qwen-VL instruction-tuning format.
    """

    def __init__(
        self,
        input_json_path: Path,
        output_json_path: Path,
        project_root: Optional[Path] = None,
    ):
        self.input_json_path = Path(input_json_path)
        self.output_json_path = Path(output_json_path)
        self.project_root = Path(project_root) if project_root is not None else None

    def load_samples(self) -> List[Dict[str, Any]]:
        if not self.input_json_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {self.input_json_path}")

        with open(self.input_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def fix_image_path(self, sample: Dict[str, Any]) -> str:
        if self.project_root is None:
            return sample["image_path"]

        filename = sample.get("image_filename")
        if filename is None:
            filename = Path(sample["image_path"]).name

        return str(self.project_root / "ds" / "img" / filename)

    @staticmethod
    def make_instruction() -> str:
        return """
You are an expert surgical vision-language assistant.

Describe this laparoscopic cholecystectomy frame using only visible evidence.
Return a cautious, uncertainty-aware surgical communication answer.

Do not hallucinate:
- instruments
- anatomy
- bleeding severity
- complications
- surgical phase
- Critical View of Safety

If something is uncertain, say it is uncertain.
"""

    @staticmethod
    def make_answer(sample: Dict[str, Any]) -> str:
        teacher_answer = sample["teacher_answer"]

        answer = {
            "visible_instruments": teacher_answer.get("visible_instruments", []),
            "visible_anatomy_or_tissue": teacher_answer.get("visible_anatomy_or_tissue", []),
            "visible_action": teacher_answer.get("visible_action", ""),
            "possible_surgical_phase": teacher_answer.get("possible_surgical_phase", ""),
            "expert_surgical_description": teacher_answer.get("expert_surgical_description", ""),
            "uncertainty_note": teacher_answer.get("uncertainty_note", ""),
        }

        return json.dumps(answer, ensure_ascii=False)

    def convert_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.fix_image_path(sample)

        return {
            "sample_id": sample["sample_id"],
            "image": image_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": self.make_instruction()},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": self.make_answer(sample)}
                    ],
                },
            ],
            "metadata": {
                "dataset": sample.get("dataset", "CholecSeg8k_sample"),
                "task": "expert_surgical_communication",
                "video_id": sample.get("video_id"),
                "sequence": sample.get("sequence"),
                "visible_classes": sample.get("visible_classes", []),
                "teacher_source": sample.get("teacher_answer", {}).get("teacher_source", ""),
            },
        }

    def build(self) -> List[Dict[str, Any]]:
        samples = self.load_samples()
        converted = [self.convert_sample(sample) for sample in samples]

        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved LoRA dataset to: {self.output_json_path}")
        print(f"[INFO] Number of samples: {len(converted)}")

        return converted