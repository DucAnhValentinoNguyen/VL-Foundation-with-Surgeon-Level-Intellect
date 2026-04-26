from pathlib import Path

from config import (
    TEST_TEACHER_JSON_PATH,
    ZERO_SHOT_OUTPUT_PATH,
    QWEN_MODEL_NAME,
)

from scripts.modeling.zero_shot import QwenZeroShotRunner


def main():
    # In Colab, change this to your Drive project path:
    # /content/drive/MyDrive/surgical_vlm
    project_root = Path.cwd()

    runner = QwenZeroShotRunner(
        model_name=QWEN_MODEL_NAME,
        input_json_path=TEST_TEACHER_JSON_PATH,
        output_json_path=ZERO_SHOT_OUTPUT_PATH,
        project_root=project_root,
        max_samples=5,
    )

    runner.run()


if __name__ == "__main__":
    main()