import json
from pathlib import Path
from typing import Dict, Any, List


class CholecystectomyTeacherLabelBuilder:
    """
    Builds expert-style surgical communication labels for laparoscopic cholecystectomy frames.

    Visual content comes only from CholecSeg8k annotations.
    Surgical literature is used only to guide safe communication style.
    """

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_samples(self) -> List[Dict[str, Any]]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input annotation file not found: {self.input_path}")

        with open(self.input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _join_items(items: List[str]) -> str:
        if not items:
            return "none"
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    @staticmethod
    def _make_anatomy_sentence(anatomy: List[str]) -> str:
        if not anatomy:
            return "No specific anatomy or tissue class is annotated apart from ignored background."

        anatomy_text = CholecystectomyTeacherLabelBuilder._join_items(anatomy)
        return f"The annotated visible anatomy or tissue classes are {anatomy_text}."

    @staticmethod
    def _make_instrument_sentence(instruments: List[str]) -> str:
        if not instruments:
            return (
                "No surgical instrument is annotated in this frame, so no tool-based action "
                "such as traction, dissection, clipping, cauterization, or adhesiolysis should be claimed."
            )

        instrument_text = CholecystectomyTeacherLabelBuilder._join_items(instruments)

        if "l-hook electrocautery" in instruments:
            return (
                f"The annotated visible instrument is {instrument_text}. "
                "In laparoscopic cholecystectomy, hook/electrocautery may be used during dissection "
                "or adhesiolysis, but the exact maneuver cannot be confirmed from a single frame alone."
            )

        if "grasper" in instruments:
            return (
                f"The annotated visible instrument is {instrument_text}. "
                "A grasper may be used for tissue or gallbladder handling, but the exact action "
                "cannot be confirmed from a single frame alone."
            )

        return (
            f"The annotated visible instrument(s) are {instrument_text}. "
            "The exact surgical maneuver cannot be confirmed from a single frame alone."
        )

    @staticmethod
    def _make_cholecystectomy_context(anatomy: List[str], instruments: List[str]) -> str:
        context_parts = []

        if "gallbladder" in anatomy:
            context_parts.append(
                "Because the gallbladder is annotated, the frame is consistent with the operative field "
                "of laparoscopic cholecystectomy."
            )

        if "cystic duct" in anatomy:
            context_parts.append(
                "The cystic duct is annotated; however, the assistant should not claim safe identification, "
                "clipping, or division unless the visual evidence and temporal context support it."
            )

        if "liver" in anatomy:
            context_parts.append(
                "The liver is annotated, which is expected near the gallbladder bed in this procedure."
            )

        if "hepatic vein" in anatomy:
            context_parts.append(
                "A hepatic vein label is present; this should be mentioned cautiously and not overinterpreted "
                "as a complication or injury."
            )

        if "blood" in anatomy:
            context_parts.append(
                "Blood is annotated, but the severity, source, or clinical significance of bleeding cannot be "
                "determined from a single frame."
            )

        if "l-hook electrocautery" in instruments:
            context_parts.append(
                "Since l-hook electrocautery is annotated, energy-device use may be relevant, but the response "
                "should avoid claiming active cauterization unless clearly visible."
            )

        if "grasper" in instruments:
            context_parts.append(
                "Since a grasper is annotated, tissue handling or retraction may be possible, but the exact "
                "action remains uncertain from one image."
            )

        if not context_parts:
            context_parts.append(
                "This frame belongs to the laparoscopic cholecystectomy domain, but the available annotation "
                "does not support a more specific procedural interpretation."
            )

        return " ".join(context_parts)

    @staticmethod
    def _make_safety_note(anatomy: List[str], instruments: List[str]) -> str:
        safety_notes = [
            "Do not infer the surgical phase from a single frame.",
            "Do not claim Critical View of Safety, clipping, cutting, duct division, or complication unless explicitly supported.",
            "Only mention structures and instruments present in the annotation.",
        ]

        if "cystic duct" not in anatomy:
            safety_notes.append(
                "Do not claim cystic duct identification because it is not annotated in this frame."
            )

        if "l-hook electrocautery" not in instruments:
            safety_notes.append(
                "Do not claim electrocautery or hook dissection because l-hook electrocautery is not annotated."
            )

        if "grasper" not in instruments:
            safety_notes.append(
                "Do not claim grasper-based retraction because a grasper is not annotated."
            )

        return " ".join(safety_notes)

    @staticmethod
    def make_teacher_answer(sample: Dict[str, Any]) -> Dict[str, Any]:
        instruments = sample.get("visible_instruments", [])
        anatomy = sample.get("visible_anatomy_or_tissue", [])

        anatomy_sentence = CholecystectomyTeacherLabelBuilder._make_anatomy_sentence(anatomy)
        instrument_sentence = CholecystectomyTeacherLabelBuilder._make_instrument_sentence(instruments)
        chole_context = CholecystectomyTeacherLabelBuilder._make_cholecystectomy_context(
            anatomy=anatomy,
            instruments=instruments,
        )
        safety_note = CholecystectomyTeacherLabelBuilder._make_safety_note(
            anatomy=anatomy,
            instruments=instruments,
        )

        expert_description = (
            f"{anatomy_sentence} "
            f"{instrument_sentence} "
            f"{chole_context} "
            "The surgical phase remains uncertain from this single frame. "
            "A safe surgical VLM assistant should describe only visible, annotation-supported evidence "
            "and explicitly communicate uncertainty."
        )

        return {
            "visible_instruments": instruments,
            "visible_anatomy_or_tissue": anatomy,
            "visible_action": instrument_sentence,
            "possible_surgical_phase": "uncertain from this single frame",
            "expert_surgical_description": expert_description,
            "uncertainty_note": safety_note,
            "teacher_source": "cholecystectomy_literature_guided_rule_teacher",
            "literature_guidance_used": [
                "laparoscopic cholecystectomy domain",
                "gallbladder and cystic pedicle awareness",
                "safe uncertainty-aware communication",
                "avoid unsupported tool/action/phase claims",
            ],
        }

    def build(self) -> List[Dict[str, Any]]:
        samples = self.load_samples()

        for sample in samples:
            sample["teacher_answer"] = self.make_teacher_answer(sample)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved literature-guided teacher labels to: {self.output_path}")
        print(f"[INFO] Total samples: {len(samples)}")

        return samples