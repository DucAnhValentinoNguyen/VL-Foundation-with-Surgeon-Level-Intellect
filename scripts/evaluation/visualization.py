import json
from pathlib import Path
from typing import List, Dict, Any


class EvaluationTableBuilder:
    """
    Creates simple markdown summaries from evaluation results.
    """

    def __init__(self, evaluation_json_path: Path):
        self.evaluation_json_path = Path(evaluation_json_path)

    def load_results(self) -> List[Dict[str, Any]]:
        with open(self.evaluation_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_summary(self) -> Dict[str, Any]:
        results = self.load_results()

        if not results:
            return {}

        n = len(results)

        def avg_bool(key):
            return sum(int(r[key]) for r in results) / n

        avg_score = sum(r["total_score_0_to_5"] for r in results) / n

        return {
            "num_samples": n,
            "json_valid_rate": avg_bool("json_valid"),
            "hallucination_free_rate": avg_bool("hallucination_free"),
            "phase_safe_rate": avg_bool("phase_safe"),
            "uncertainty_present_rate": avg_bool("uncertainty_present"),
            "expert_style_rate": avg_bool("expert_style"),
            "average_score_0_to_5": avg_score,
        }

    def print_markdown_table(self):
        summary = self.build_summary()

        if not summary:
            print("No results available.")
            return

        print("| Metric | Value |")
        print("|---|---:|")
        print(f"| Samples | {summary['num_samples']} |")
        print(f"| JSON valid rate | {summary['json_valid_rate']:.2f} |")
        print(f"| Hallucination-free rate | {summary['hallucination_free_rate']:.2f} |")
        print(f"| Phase-safe rate | {summary['phase_safe_rate']:.2f} |")
        print(f"| Uncertainty-present rate | {summary['uncertainty_present_rate']:.2f} |")
        print(f"| Expert-style rate | {summary['expert_style_rate']:.2f} |")
        print(f"| Average score | {summary['average_score_0_to_5']:.2f}/5 |")