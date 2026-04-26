from config import (
    TRAIN_TEACHER_JSON_PATH,
    VAL_TEACHER_JSON_PATH,
    TEST_TEACHER_JSON_PATH,
    LORA_TRAIN_DATA_PATH,
    LORA_VAL_DATA_PATH,
    LORA_TEST_DATA_PATH,
)

from scripts.modeling.lora_dataset import LoraDatasetBuilder


def main():
    for input_path, output_path in [
        (TRAIN_TEACHER_JSON_PATH, LORA_TRAIN_DATA_PATH),
        (VAL_TEACHER_JSON_PATH, LORA_VAL_DATA_PATH),
        (TEST_TEACHER_JSON_PATH, LORA_TEST_DATA_PATH),
    ]:
        builder = LoraDatasetBuilder(
            input_json_path=input_path,
            output_json_path=output_path,
        )
        builder.build()


if __name__ == "__main__":
    main()