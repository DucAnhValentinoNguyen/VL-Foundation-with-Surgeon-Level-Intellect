from config import (
    EXPERT_ANNOTATION_PATH,
    TEACHER_ANNOTATION_PATH,
    TRAIN_TEACHER_JSON_PATH,
    VAL_TEACHER_JSON_PATH,
    TEST_TEACHER_JSON_PATH,
)
from scripts.data.teacher_labeler import CholecystectomyTeacherLabelBuilder
from scripts.data.annotation_builder import ExpertCommunicationAnnotationBuilder


def main():
    builder = CholecystectomyTeacherLabelBuilder(
        input_path=EXPERT_ANNOTATION_PATH,
        output_path=TEACHER_ANNOTATION_PATH,
    )

    samples = builder.build()

    ExpertCommunicationAnnotationBuilder.split_by_video_id(
        samples=samples,
        train_path=TRAIN_TEACHER_JSON_PATH,
        val_path=VAL_TEACHER_JSON_PATH,
        test_path=TEST_TEACHER_JSON_PATH,
    )


if __name__ == "__main__":
    main()