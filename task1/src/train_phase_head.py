# Training and evaluation utilities for phase recognition:
# - compute frame-wise classification loss,
# - evaluate the model on the test split using loss / accuracy / macro F1,
# - train the model epoch by epoch and save training history and the final checkpoint.

import json
import os
import torch
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from phase_data import TaskAPhaseDataset, collate_fn
from phase_model import TaskAQwen25VL
from build_gt_jsonl import process_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "results"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

num_epochs = 8
lr = 1e-4
num_phase_classes = 7
batch_size = 1
max_frames_per_video = 90
device = "cuda" if torch.cuda.is_available() else "cpu"

grad_clip_norm = 1.0
adam_eps = 1e-6


def compute_loss(phase_logits, phase_labels, num_phase_classes=7):
    logits = phase_logits[0]

    valid_indices = [
        i for i, x in enumerate(phase_labels)
        if isinstance(x, int) and 0 <= x < num_phase_classes
    ]

    if len(valid_indices) == 0:
        return None

    valid_logits = logits[valid_indices]
    valid_labels = torch.tensor(
        [phase_labels[i] for i in valid_indices],
        dtype=torch.long,
        device=logits.device
    )

    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(valid_logits, valid_labels)


def evaluate_phase_model(model, loader, num_phase_classes=7):
    model.eval()
    if hasattr(model, "backbone"):
        model.backbone.eval()

    total_loss = 0.0
    steps = 0

    all_gt = []
    all_pred = []

    with torch.no_grad():
        for sample in loader:
            phase_logits = model(sample["frames"])

            loss = compute_loss(
                phase_logits,
                sample["phase_labels"],
                num_phase_classes=num_phase_classes
            )

            if loss is not None:
                total_loss += loss.item()
                steps += 1

            logits = phase_logits[0]  # [T, C]
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()

            valid_indices = [
                i for i, x in enumerate(sample["phase_labels"])
                if isinstance(x, int) and 0 <= x < num_phase_classes
            ]

            for i in valid_indices:
                all_gt.append(int(sample["phase_labels"][i]))
                all_pred.append(int(pred_ids[i]))

    test_loss = total_loss / max(steps, 1)
    test_accuracy = accuracy_score(all_gt, all_pred) if len(all_gt) > 0 else 0.0
    test_macro_f1 = f1_score(all_gt, all_pred, average="macro", zero_division=0) if len(all_gt) > 0 else 0.0

    return {
        "test_loss": float(test_loss),
        "test_framewise_accuracy": float(test_accuracy),
        "test_framewise_macro_f1": float(test_macro_f1),
    }


def train(model, train_loader, test_loader, optimizer, save_dir, save_name, epochs=8, num_phase_classes=7):
    os.makedirs(save_dir, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable_params))

    global_step = 0
    history = []

    best_metric = -float("inf")
    best_epoch = -1
    best_state_dict = None
    best_eval_metrics = None
    best_train_loss = None

    for epoch in range(epochs):
        model.train()
        if getattr(model, "freeze_backbone", False):
            model.backbone.eval()

        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for sample in pbar:
            global_step += 1

            try:
                phase_logits = model(sample["frames"])
            except Exception as e:
                print("\n[FORWARD FAIL]")
                print("video:", sample["video_path"])
                print("frame_indices:", sample["frame_indices"])
                raise e

            if torch.isnan(phase_logits).any() or torch.isinf(phase_logits).any():
                print("\n[BAD LOGITS]")
                print("video:", sample["video_path"])
                print("frame_indices:", sample["frame_indices"])
                print("phase_labels:", sample["phase_labels"])
                raise ValueError("phase_logits invalid")

            loss = compute_loss(
                phase_logits,
                sample["phase_labels"],
                num_phase_classes=num_phase_classes
            )

            if loss is None:
                print("\n[SKIP] no valid labels:", sample["video_path"])
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                print("\n[BAD LOSS]")
                print("video:", sample["video_path"])
                print("phase_labels:", sample["phase_labels"])
                print("logits min/max:", phase_logits.min().item(), phase_logits.max().item())
                raise ValueError("loss invalid")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            pbar.set_postfix(
                loss=f"{running_loss / max(steps,1):.4f}",
                last=f"{loss.item():.4f}",
                step=global_step
            )

        train_loss = running_loss / max(steps, 1)

        eval_metrics = evaluate_phase_model(
            model=model,
            loader=test_loader,
            num_phase_classes=num_phase_classes
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "test_loss": eval_metrics["test_loss"],
            "test_framewise_accuracy": eval_metrics["test_framewise_accuracy"],
            "test_framewise_macro_f1": eval_metrics["test_framewise_macro_f1"],
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={eval_metrics['test_loss']:.4f} | "
            f"test_acc={eval_metrics['test_framewise_accuracy']:.4f} | "
            f"test_macro_f1={eval_metrics['test_framewise_macro_f1']:.4f}"
        )

        with open(os.path.join(save_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        current_metric = eval_metrics["test_framewise_macro_f1"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_train_loss = float(train_loss)
            best_eval_metrics = copy.deepcopy(eval_metrics)
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("No valid best checkpoint was selected during training.")

    ckpt = {
        "epoch": best_epoch,
        "model_state_dict": best_state_dict,
        "best_metric_name": "test_framewise_macro_f1",
        "best_metric_value": best_metric,
        "train_loss": best_train_loss,
        "test_loss": best_eval_metrics["test_loss"],
        "test_framewise_accuracy": best_eval_metrics["test_framewise_accuracy"],
        "test_framewise_macro_f1": best_eval_metrics["test_framewise_macro_f1"],
        "train_history": history,
    }

    torch.save(ckpt, os.path.join(save_dir, save_name))

    print(f"Training done. Best epoch: {best_epoch}")
    print(f"Best test_framewise_macro_f1: {best_metric:.4f}")
    print("Saved to:", os.path.join(save_dir, save_name))


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
        save_name="qwen25vl_frozen.pt",
        epochs=num_epochs,
        num_phase_classes=num_phase_classes
    )



