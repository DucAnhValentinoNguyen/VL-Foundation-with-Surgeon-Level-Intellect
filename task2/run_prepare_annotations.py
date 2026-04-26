from config import (
    IMAGE_DIR,
    ANNOTATION_DIR,
    META_PATH,
    EXPERT_ANNOTATION_PATH,
    TRAIN_JSON_PATH,
    VAL_JSON_PATH,
    TEST_JSON_PATH,
)

from scripts.data.annotation_builder import ExpertCommunicationAnnotationBuilder


def main():
    builder = ExpertCommunicationAnnotationBuilder(
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        meta_path=META_PATH,
        output_path=EXPERT_ANNOTATION_PATH,
    )

    samples = builder.build()

    builder.split_by_video_id(
        samples=samples,
        train_path=TRAIN_JSON_PATH,
        val_path=VAL_JSON_PATH,
        test_path=TEST_JSON_PATH,
    )


if __name__ == "__main__":
    main()