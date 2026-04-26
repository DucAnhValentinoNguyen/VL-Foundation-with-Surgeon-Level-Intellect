import os
import torch
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from train_phase_frozen_backbone import train
from phase_data import TaskAPhaseDataset, collate_fn
from phase_model import TaskAQwen25VL
from build_gt_jsonl import process_split



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "results"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

num_epochs = 8
lr = 5e-6
num_phase_classes = 7
batch_size = 1
max_frames_per_video = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

grad_clip_norm = 1.0
adam_eps = 1e-6

if __name__ == "__main__":

    train_dir = DATA_DIR / "training"
    test_dir = DATA_DIR / "testing"
    gt_name = "gt_10s.jsonl"
    train_jsonl = train_dir / gt_name
    test_jsonl = test_dir / gt_name

    process_split(test_dir, gt_name=gt_name, sample_stride_sec=10)
    process_split(train_dir, gt_name=gt_name, sample_stride_sec=10)

    train_dataset = TaskAPhaseDataset(
        train_jsonl,
        max_frames_per_video=max_frames_per_video,
        num_phase_classes=num_phase_classes,
        validate=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_dataset = TaskAPhaseDataset(
        test_jsonl,
        max_frames_per_video=max_frames_per_video,
        num_phase_classes=num_phase_classes,
        validate=True,
        random_offset_sec=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = TaskAQwen25VL(
        model_name=MODEL_NAME,
        num_phase_classes=num_phase_classes,
        use_lora=False,
        freeze_backbone=True,
        backbone_dtype=torch.float32,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        eps=adam_eps
    )

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        save_dir=SAVE_DIR,
        save_name="qwen25vl_lora.pt",
        epochs=num_epochs,
        num_phase_classes=num_phase_classes
    )




