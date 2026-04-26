from pathlib import Path

# ============================================================
# Project root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent


# ============================================================
# Dataset paths
# ============================================================

DATASET_DIR = ROOT_DIR / "ds"
IMAGE_DIR = DATASET_DIR / "img"
ANNOTATION_DIR = DATASET_DIR / "ann"
META_PATH = DATASET_DIR / "meta.json"


# ============================================================
# Output directories
# ============================================================

OUTPUT_DIR = ROOT_DIR / "outputs"
ANNOTATION_OUTPUT_DIR = OUTPUT_DIR / "annotations"
ZERO_SHOT_OUTPUT_DIR = OUTPUT_DIR / "zero_shot_predictions"
LORA_OUTPUT_DIR = OUTPUT_DIR / "lora_data"
LORA_ADAPTER_OUTPUT_DIR = OUTPUT_DIR / "lora_adapter"
LORA_PREDICTION_OUTPUT_DIR = OUTPUT_DIR / "lora_predictions"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"


# ============================================================
# Annotation output paths
# ============================================================

EXPERT_ANNOTATION_PATH = ANNOTATION_OUTPUT_DIR / "expert_communication_annotations.json"

TRAIN_JSON_PATH = ANNOTATION_OUTPUT_DIR / "train_expert_comm.json"
VAL_JSON_PATH = ANNOTATION_OUTPUT_DIR / "val_expert_comm.json"
TEST_JSON_PATH = ANNOTATION_OUTPUT_DIR / "test_expert_comm.json"

TEACHER_ANNOTATION_PATH = ANNOTATION_OUTPUT_DIR / "teacher_expert_communication_annotations.json"

TRAIN_TEACHER_JSON_PATH = ANNOTATION_OUTPUT_DIR / "train_teacher_expert_comm.json"
VAL_TEACHER_JSON_PATH = ANNOTATION_OUTPUT_DIR / "val_teacher_expert_comm.json"
TEST_TEACHER_JSON_PATH = ANNOTATION_OUTPUT_DIR / "test_teacher_expert_comm.json"


# ============================================================
# LoRA dataset paths
# ============================================================

LORA_TRAIN_DATA_PATH = LORA_OUTPUT_DIR / "qwen_lora_train.json"
LORA_VAL_DATA_PATH = LORA_OUTPUT_DIR / "qwen_lora_val.json"
LORA_TEST_DATA_PATH = LORA_OUTPUT_DIR / "qwen_lora_test.json"


# ============================================================
# Zero-shot prediction paths
# ============================================================

ZERO_SHOT_OUTPUT_PATH = ZERO_SHOT_OUTPUT_DIR / "qwen_zero_shot_predictions.json"


# ============================================================
# LoRA prediction paths
# ============================================================

LORA_PREDICTION_PATH = LORA_PREDICTION_OUTPUT_DIR / "qwen_lora_predictions.json"


# ============================================================
# Evaluation paths
# ============================================================

ZERO_SHOT_EVAL_PATH = EVALUATION_OUTPUT_DIR / "zero_shot_evaluation.json"
ZERO_SHOT_EVAL_TABLE_PATH = EVALUATION_OUTPUT_DIR / "zero_shot_evaluation_table.csv"

LORA_EVAL_PATH = EVALUATION_OUTPUT_DIR / "lora_evaluation.json"
LORA_EVAL_TABLE_PATH = EVALUATION_OUTPUT_DIR / "lora_evaluation_table.csv"

EVALUATION_OUTPUT_PATH = EVALUATION_OUTPUT_DIR / "evaluation_results.json"


# ============================================================
# Model choice
# ============================================================

# Start with 3B in Colab. 7B may be too heavy on free GPU.
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ============================================================
# Actual video IDs found in downloaded DatasetNinja CholecSeg8k sample
# ============================================================

VIDEO_FRAME_COUNTS = {
    0: 25,
    1: 160,
    2: 191,
    3: 74,
    4: 92,
    5: 89,
    6: 43,
    7: 146,
    8: 112,
    9: 31,
}


# ============================================================
# Video-level split
# ============================================================

TRAIN_VIDEOS = {0, 1, 2, 3, 4, 5, 6}
VAL_VIDEOS = {7}
TEST_VIDEOS = {8, 9}


# ============================================================
# Class grouping based on meta.json
# ============================================================

IGNORE_CLASSES = {
    "black background"
}

INSTRUMENT_CLASSES = {
    "grasper",
    "l-hook electrocautery",
}

ANATOMY_OR_TISSUE_CLASSES = {
    "abdominal wall",
    "blood",
    "connective tissue",
    "cystic duct",
    "fat",
    "gallbladder",
    "gastrointestinal tract",
    "hepatic vein",
    "liver",
    "liver ligament",
}