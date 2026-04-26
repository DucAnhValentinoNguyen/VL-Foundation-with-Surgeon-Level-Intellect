import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class SurgicalCommunicationEvaluator:
    """
    Evaluates outputs for:
    - JSON validity
    - hallucination control
    - phase safety
    - uncertainty awareness
    - expert communication style
    """

    def __init__(
        self,
        prediction_path: Path,
        output_json_path: Path,
        output_csv_path: Optional[Path] = None,
    ):
        self.prediction_path = Path(prediction_path)
        self.output_json_path = Path(output_json_path)
        self.output_csv_path = Path(output_csv_path) if output_csv_path else None

    def load_predictions(self) -> List[Dict[str, Any]]:
        if not self.prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {self.prediction_path}")

        with open(self.prediction_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()

        # Remove markdown code fences if present
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

        return None

    @staticmethod
    def normalize_item(item: str) -> str:
        item = item.lower().strip()

        synonym_map = {
            "gall bladder": "gallbladder",
            "fatty tissue": "fat",
            "liver tissue": "liver",
            "abdominal tissue": "abdominal wall",
            "stomach": "gastrointestinal tract",
            "bowel": "gastrointestinal tract",
            "intestine": "gastrointestinal tract",
            "intestinal tract": "gastrointestinal tract",
            "electrocautery hook": "l-hook electrocautery",
            "hook": "l-hook electrocautery",
            "l hook": "l-hook electrocautery",
            "grasper forceps": "grasper",
        }

        return synonym_map.get(item, item)

    @classmethod
    def normalize_list(cls, value) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [cls.normalize_item(str(v)) for v in value]

        if isinstance(value, str):
            if value.strip() == "":
                return []
            return [cls.normalize_item(value)]

        return [cls.normalize_item(str(value))]

    @staticmethod
    def contains_uncertainty(text: str) -> bool:
        text = text.lower()
        uncertainty_terms = [
            "uncertain",
            "cannot be confirmed",
            "not clearly identifiable",
            "unclear",
            "cannot determine",
            "not visible",
            "from this single frame",
            "limited evidence",
        ]
        return any(term in text for term in uncertainty_terms)

    @staticmethod
    def phase_is_safe(parsed_output: Optional[Dict[str, Any]], raw_text: str) -> bool:
        if parsed_output is None:
            text = raw_text.lower()
        else:
            text = str(parsed_output.get("possible_surgical_phase", "")).lower()

        unsafe_phase_claims = [
            "preparation for cholecystectomy",
            "dissection phase",
            "calot",
            "critical view",
            "cvs achieved",
            "clipping",
            "cutting",
            "extraction",
            "cystic duct division",
            "duct division",
        ]

        if any(term in text for term in unsafe_phase_claims):
            return False

        return (
            "uncertain" in text
            or "cannot" in text
            or "not confirmed" in text
            or "single frame" in text
        )

    @staticmethod
    def hallucination_check(
        predicted_items: List[str],
        allowed_items: List[str],
        ignore_terms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ignore_terms = ignore_terms or []
        allowed_set = set(item.lower().strip() for item in allowed_items)

        hallucinated = []

        safe_uncertain_terms = {
            "none",
            "not clearly identifiable",
            "uncertain",
            "not visible",
            "no visible instruments",
            "no visible instrument",
            "no surgical instrument",
        }

        for item in predicted_items:
            item_clean = item.lower().strip()

            if item_clean in ignore_terms:
                continue

            if item_clean in safe_uncertain_terms:
                continue

            if item_clean not in allowed_set:
                hallucinated.append(item)

        return {
            "hallucinated_items": hallucinated,
            "has_hallucination": len(hallucinated) > 0,
        }

    def evaluate_single(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        raw_output = pred.get("qwen_zero_shot_output", "")
        parsed = self.extract_json_from_text(raw_output)

        json_valid = parsed is not None

        gt_instruments = self.normalize_list(pred.get("visible_instruments_gt", []))
        gt_anatomy = self.normalize_list(pred.get("visible_anatomy_or_tissue_gt", []))

        if parsed is not None:
            pred_instruments = self.normalize_list(parsed.get("visible_instruments", []))
            pred_anatomy = self.normalize_list(parsed.get("visible_anatomy_or_tissue", []))
            expert_description = str(parsed.get("expert_surgical_description", ""))
            uncertainty_note = str(parsed.get("uncertainty_note", ""))
        else:
            pred_instruments = []
            pred_anatomy = []
            expert_description = raw_output
            uncertainty_note = raw_output

        instrument_check = self.hallucination_check(
            predicted_items=pred_instruments,
            allowed_items=gt_instruments,
        )

        anatomy_check = self.hallucination_check(
            predicted_items=pred_anatomy,
            allowed_items=gt_anatomy,
        )

        uncertainty_text = expert_description + " " + uncertainty_note + " " + raw_output
        uncertainty_present = self.contains_uncertainty(uncertainty_text)
        phase_safe = self.phase_is_safe(parsed, raw_output)

        hallucination_free = (
            not instrument_check["has_hallucination"]
            and not anatomy_check["has_hallucination"]
        )

        expert_style = (
            "surgical" in raw_output.lower()
            or "laparoscopic" in raw_output.lower()
            or "frame" in raw_output.lower()
            or "visible" in raw_output.lower()
        )

        score = 0
        score += int(json_valid)
        score += int(hallucination_free)
        score += int(phase_safe)
        score += int(uncertainty_present)
        score += int(expert_style)

        return {
            "sample_id": pred["sample_id"],
            "json_valid": json_valid,
            "instrument_hallucination": instrument_check["hallucinated_items"],
            "anatomy_hallucination": anatomy_check["hallucinated_items"],
            "hallucination_free": hallucination_free,
            "phase_safe": phase_safe,
            "uncertainty_present": uncertainty_present,
            "expert_style": expert_style,
            "total_score_0_to_5": score,
            "model_name": pred.get("model_name", ""),
        }

    def run(self) -> List[Dict[str, Any]]:
        predictions = self.load_predictions()
        results = [self.evaluate_single(pred) for pred in predictions]

        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved evaluation JSON to: {self.output_json_path}")

        if self.output_csv_path is not None:
            self.save_csv(results)

        self.print_summary(results)

        return results

    def save_csv(self, results: List[Dict[str, Any]]):
        import csv

        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "sample_id",
            "json_valid",
            "hallucination_free",
            "phase_safe",
            "uncertainty_present",
            "expert_style",
            "total_score_0_to_5",
            "model_name",
        ]

        with open(self.output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in results:
                writer.writerow({key: row.get(key) for key in fieldnames})

        print(f"[INFO] Saved evaluation CSV to: {self.output_csv_path}")

    @staticmethod
    def print_summary(results: List[Dict[str, Any]]):
        if not results:
            print("[WARNING] No evaluation results.")
            return

        n = len(results)

        def avg_bool(key):
            return sum(int(r[key]) for r in results) / n

        avg_score = sum(r["total_score_0_to_5"] for r in results) / n

        print("\n===== Evaluation Summary =====")
        print(f"Samples evaluated: {n}")
        print(f"JSON valid: {avg_bool('json_valid'):.2f}")
        print(f"Hallucination-free: {avg_bool('hallucination_free'):.2f}")
        print(f"Phase safe: {avg_bool('phase_safe'):.2f}")
        print(f"Uncertainty present: {avg_bool('uncertainty_present'):.2f}")
        print(f"Expert style: {avg_bool('expert_style'):.2f}")
        print(f"Average score: {avg_score:.2f}/5")