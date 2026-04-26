from pathlib import Path
from typing import Optional

from scripts.modeling.zero_shot import QwenZeroShotRunner
from scripts.evaluation.evaluator import SurgicalCommunicationEvaluator
from scripts.modeling.lora_dataset import LoraDatasetBuilder

class SurgicalVLMExperiment:
    """
    High-level experiment runner for Task 5:
    Expert-Level Surgical Communication.
    """

    def __init__(
        self,
        project_root: Path,
        model_name: str,
        test_json_path: Path,
        zero_shot_output_path: Path,
        eval_json_path: Path,
        eval_csv_path: Path,
    ):
        self.project_root = Path(project_root)
        self.model_name = model_name
        self.test_json_path = Path(test_json_path)
        self.zero_shot_output_path = Path(zero_shot_output_path)
        self.eval_json_path = Path(eval_json_path)
        self.eval_csv_path = Path(eval_csv_path)

    def run_zero_shot(self, max_samples: int = 5):
        runner = QwenZeroShotRunner(
            model_name=self.model_name,
            input_json_path=self.test_json_path,
            output_json_path=self.zero_shot_output_path,
            project_root=self.project_root,
            max_samples=max_samples,
        )

        return runner.run()

    def evaluate_zero_shot(self):
        evaluator = SurgicalCommunicationEvaluator(
            prediction_path=self.zero_shot_output_path,
            output_json_path=self.eval_json_path,
            output_csv_path=self.eval_csv_path,
        )

        return evaluator.run()

    def prepare_lora_dataset(
        self,
        train_json_path: Path,
        val_json_path: Path,
        test_json_path: Path,
        lora_train_output_path: Path,
        lora_val_output_path: Path,
        lora_test_output_path: Path,
    ):
        train_builder = LoraDatasetBuilder(
            input_json_path=train_json_path,
            output_json_path=lora_train_output_path,
            project_root=self.project_root,
        )

        val_builder = LoraDatasetBuilder(
            input_json_path=val_json_path,
            output_json_path=lora_val_output_path,
            project_root=self.project_root,
        )

        test_builder = LoraDatasetBuilder(
            input_json_path=test_json_path,
            output_json_path=lora_test_output_path,
            project_root=self.project_root,
        )

        train_data = train_builder.build()
        val_data = val_builder.build()
        test_data = test_builder.build()

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }