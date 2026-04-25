import os
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLModel
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import time


train_jsonl = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/training/finetune_long_video_10s.jsonl"

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
save_dir = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/taskA_qwen25vl_lora"

num_epochs = 15
lr = 5e-6
num_phase_classes = 7
batch_size = 1
max_frames_per_video = 2  
device = "cuda" if torch.cuda.is_available() else "cpu"


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


grad_clip_norm = 0.5
adam_eps = 1e-6
# =========================

os.makedirs(save_dir, exist_ok=True)
torch.autograd.set_detect_anomaly(True)


# -------------------------
# Utils
# -------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def can_open_video(video_path):
    if not os.path.exists(video_path):
        return False
    cap = cv2.VideoCapture(video_path)
    ok = cap.isOpened()
    cap.release()
    return ok




def read_frames_rgb(video_path, frame_indices, max_retries=3, sleep_sec=1.0):
    last_error = None

    for attempt in range(max_retries):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            last_error = RuntimeError(f"Failed to open video: {video_path} | attempt={attempt+1}")
            cap.release()
            time.sleep(sleep_sec)
            continue

        frames = []
        ok_all = True

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                ok_all = False
                last_error = RuntimeError(
                    f"Failed to read frame {frame_index} from {video_path} | attempt={attempt+1}"
                )
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if ok_all:
            return frames

        time.sleep(sleep_sec)

    raise last_error


# -------------------------
# Dataset
# -------------------------
class TaskADataset(Dataset):
    def __init__(self, jsonl_path, max_frames_per_video=None, num_phase_classes=7):
        raw_rows = load_jsonl(jsonl_path)
        self.rows = []
        self.max_frames_per_video = max_frames_per_video
        self.num_phase_classes = num_phase_classes

        for row in raw_rows:
            video_path = row["video"]

            if not can_open_video(video_path):
                print("[SKIP] bad video:", video_path)
                continue

            frame_indices = row["frame_indices"]
            phase_labels = row["phase_labels"]

            if len(frame_indices) != len(phase_labels):
                print("[SKIP] length mismatch:", video_path)
                continue

            valid_count = sum(
                1 for x in phase_labels
                if isinstance(x, int) and 0 <= x < self.num_phase_classes
            )
            if valid_count == 0:
                print("[SKIP] no valid phase labels:", video_path)
                continue

            self.rows.append(row)

        print("num usable rows:", len(self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        video_path = row["video"]
        frame_indices = row["frame_indices"]
        phase_labels = row["phase_labels"]

        if self.max_frames_per_video is not None:
            frame_indices = frame_indices[:self.max_frames_per_video]
            phase_labels = phase_labels[:self.max_frames_per_video]

        frames = read_frames_rgb(video_path, frame_indices)

        return {
            "video_path": video_path,
            "frames": frames,
            "frame_indices": frame_indices,
            "phase_labels": phase_labels,
        }


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


# -------------------------
# Model
# -------------------------
class TaskAQwen25VLLoRA(nn.Module):
    def __init__(
        self,
        model_name,
        num_phase_classes=7,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)

        backbone = Qwen2_5_VLModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.backbone = get_peft_model(backbone, lora_config)

        hidden_size = self.backbone.config.text_config.hidden_size

        self.temporal = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.feat_norm = nn.LayerNorm(hidden_size)
        self.phase_head = nn.Linear(hidden_size, num_phase_classes)

        self.backbone.print_trainable_parameters()

    def encode_one_frame(self, frame_rgb):
        device = next(self.parameters()).device

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_rgb},
                    {"type": "text", "text": "phase classification"}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        outputs = self.backbone(**inputs, return_dict=True)
        feat = outputs.last_hidden_state.mean(dim=1)[0]

        if torch.isnan(feat).any() or torch.isinf(feat).any():
            raise ValueError("feature has NaN/Inf inside encode_one_frame")

        return feat.float()

    def forward(self, frames):
        feats = []
        for frame_rgb in frames:
            feat = self.encode_one_frame(frame_rgb)
            feats.append(feat)

        feats = torch.stack(feats, dim=0).unsqueeze(0)   # [1, T, D]
        feats = self.feat_norm(feats)

        if torch.isnan(feats).any() or torch.isinf(feats).any():
            raise ValueError("feats has NaN/Inf after feat_norm")
        temporal_out, _ = self.temporal(feats) 
        phase_logits = self.phase_head(temporal_out)            # [1, T, 7]

        if torch.isnan(phase_logits).any() or torch.isinf(phase_logits).any():
            raise ValueError("phase_logits has NaN/Inf")

        return phase_logits


# -------------------------
# Loss
# -------------------------
def compute_loss(phase_logits, phase_labels, num_phase_classes=7):
    logits = phase_logits[0]  # [T, 7]

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
    loss = loss_fn(valid_logits, valid_labels)
    return loss


# -------------------------
# Train
# -------------------------
def train():
    train_dataset = TaskADataset(
        train_jsonl,
        max_frames_per_video=max_frames_per_video,
        num_phase_classes=num_phase_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = TaskAQwen25VLLoRA(
        model_name=model_name,
        num_phase_classes=num_phase_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        eps=adam_eps
    )

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

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

            bad_grad = False
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print("\n[BAD GRAD]", name)
                        print("video:", sample["video_path"])
                        bad_grad = True
                        break
            if bad_grad:
                raise ValueError("gradient invalid")

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            optimizer.step()

            bad_param = False
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                        print("\n[BAD PARAM AFTER STEP]", name)
                        print("video:", sample["video_path"])
                        bad_param = True
                        break
            if bad_param:
                raise ValueError("parameter invalid after step")

            running_loss += loss.item()
            steps += 1

            pbar.set_postfix(
                loss=f"{running_loss / max(steps,1):.4f}",
                last=f"{loss.item():.4f}",
                step=global_step
            )

        train_loss = running_loss / max(steps, 1)

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f}")

    ckpt = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
    }

    #torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))
    torch.save(ckpt, os.path.join(save_dir, "last.pt"))

    print("Training done.")
    print("Saved to:", save_dir)


if __name__ == "__main__":
    train()