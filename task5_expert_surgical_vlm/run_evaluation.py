from config import (
    ZERO_SHOT_OUTPUT_PATH,
    ZERO_SHOT_EVAL_PATH,
    ZERO_SHOT_EVAL_TABLE_PATH,
)

from scripts.evaluation.evaluator import SurgicalCommunicationEvaluator


def main():
    evaluator = SurgicalCommunicationEvaluator(
        prediction_path=ZERO_SHOT_OUTPUT_PATH,
        output_json_path=ZERO_SHOT_EVAL_PATH,
        output_csv_path=ZERO_SHOT_EVAL_TABLE_PATH,
    )

    evaluator.run()


if __name__ == "__main__":
    main()