import os
import json
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLModel
from peft import LoraConfig, get_peft_model
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


train_jsonl = "/content/drive/MyDrive/CholecTrack20/CholecTrack20_15min_clips/training/finetune_long_video_10s.jsonl"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
save_dir = "/content/drive/MyDrive/CholecTrack20/taskB_qwen25vl_lora"

num_epochs = 20
lr = 5e-6
batch_size = 1
max_frames_per_video = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

num_phase_classes = 7
num_instrument_classes = 7
num_operator_classes = 4
num_slots = 3

# LoRA
lora_r = 4
lora_alpha = 8
lora_dropout = 0.05

# loss weight
lambda_phase = 1.0
lambda_obj = 1.0
lambda_instr = 1.0
lambda_op = 1.0
lambda_bbox_l1 = 2.0
lambda_bbox_iou = 2.0
lambda_intraop = 1.0
lambda_intracorp = 1.0

# matching cost weight
match_cost_bbox = 2.0
match_cost_instr = 1.0
match_cost_op = 1.0
match_cost_intraop = 0.5
match_cost_intracorp = 0.5

# stablize
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


def infer_track_vocab_sizes(rows):
    max_intraop = 0
    max_intracorp = 0
    for row in rows:
        for frame in row["anchor_targets"]:
            for obj in frame["objects"]:
                max_intraop = max(max_intraop, int(obj["intraoperative_track"]))
                max_intracorp = max(max_intracorp, int(obj["intracorporeal_track"]))
    return max_intraop + 1, max_intracorp + 1


def bbox_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou_xywh(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(box1)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(box2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_area_xyxy([ax1, ay1, ax2, ay2])
    area_b = box_area_xyxy([bx1, by1, bx2, by2])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_iou_xywh_torch(box1, box2):
    """
    box1, box2: [4] in xywh
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(w1, min=0.0) * torch.clamp(h1, min=0.0)
    area2 = torch.clamp(w2, min=0.0) * torch.clamp(h2, min=0.0)
    union = area1 + area2 - inter_area

    iou = inter_area / (union + 1e-8)
    return iou


# -------------------------
# Dataset
# -------------------------
class TaskBDataset(Dataset):
    def __init__(self, jsonl_path, max_frames_per_video=None):
        raw_rows = load_jsonl(jsonl_path)
        self.rows = []
        self.max_frames_per_video = max_frames_per_video

        for row in raw_rows:
            video_path = row["video"]

            if not can_open_video(video_path):
                print("[SKIP] bad video:", video_path)
                continue

            frame_indices = row["frame_indices"]
            anchor_targets = row["anchor_targets"]

            if len(frame_indices) != len(anchor_targets):
                print("[SKIP] length mismatch:", video_path)
                continue

            self.rows.append(row)

        print("num usable rows:", len(self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        video_path = row["video"]
        frame_indices = row["frame_indices"]
        anchor_targets = row["anchor_targets"]

        if self.max_frames_per_video is not None:
            frame_indices = frame_indices[:self.max_frames_per_video]
            anchor_targets = anchor_targets[:self.max_frames_per_video]

        frames = read_frames_rgb(video_path, frame_indices)

        return {
            "video_path": video_path,
            "frames": frames,                   # len=T
            "frame_indices": frame_indices,     # len=T
            "anchor_targets": anchor_targets,   # len=T
        }


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


# -------------------------
# Model
# -------------------------
class TaskBQwen25VLLoRA(nn.Module):
    def __init__(
        self,
        model_name,
        num_phase_classes,
        num_instrument_classes,
        num_operator_classes,
        num_intraop_classes,
        num_intracorp_classes,
        num_slots=3,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
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
            target_modules=["q_proj", "v_proj"],
        )
        self.backbone = get_peft_model(backbone, lora_config)

        hidden_size = self.backbone.config.text_config.hidden_size
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_instrument_classes = num_instrument_classes
        self.num_operator_classes = num_operator_classes
        self.num_intraop_classes = num_intraop_classes
        self.num_intracorp_classes = num_intracorp_classes

        self.feat_norm = nn.LayerNorm(hidden_size)

        self.temporal = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.phase_head = nn.Linear(hidden_size, num_phase_classes)
        self.obj_head = nn.Linear(hidden_size, num_slots)
        self.instr_head = nn.Linear(hidden_size, num_slots * num_instrument_classes)
        self.op_head = nn.Linear(hidden_size, num_slots * num_operator_classes)
        self.bbox_head = nn.Linear(hidden_size, num_slots * 4)
        self.intraop_head = nn.Linear(hidden_size, num_slots * num_intraop_classes)
        self.intracorp_head = nn.Linear(hidden_size, num_slots * num_intracorp_classes)

        self.backbone.print_trainable_parameters()

    def encode_one_frame(self, frame_rgb):
        device = next(self.parameters()).device

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_rgb},
                    {"type": "text", "text": "surgical tool and phase understanding"}
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

        temporal_out, _ = self.temporal(feats)           # [1, T, D]

        phase_logits = self.phase_head(temporal_out)     # [1, T, P]
        obj_logits = self.obj_head(temporal_out)         # [1, T, K]

        t = temporal_out.shape[1]
        k = self.num_slots

        instr_logits = self.instr_head(temporal_out).view(1, t, k, self.num_instrument_classes)
        op_logits = self.op_head(temporal_out).view(1, t, k, self.num_operator_classes)
        bbox_pred = self.bbox_head(temporal_out).view(1, t, k, 4).sigmoid()
        intraop_logits = self.intraop_head(temporal_out).view(1, t, k, self.num_intraop_classes)
        intracorp_logits = self.intracorp_head(temporal_out).view(1, t, k, self.num_intracorp_classes)

        return {
            "phase_logits": phase_logits,
            "obj_logits": obj_logits,
            "instr_logits": instr_logits,
            "op_logits": op_logits,
            "bbox_pred": bbox_pred,
            "intraop_logits": intraop_logits,
            "intracorp_logits": intracorp_logits,
        }


# -------------------------
# Matching
# -------------------------
def build_matching_cost_for_one_t(
    instr_logits_t,      # [K, C_instr]
    op_logits_t,         # [K, C_op]
    bbox_pred_t,         # [K, 4]
    intraop_logits_t,    # [K, C_intraop]
    intracorp_logits_t,  # [K, C_intracorp]
    gt_objects           # list[dict], len=M
):
    k = instr_logits_t.shape[0]
    m = len(gt_objects)

    if m == 0:
        return None

    cost = torch.zeros((k, m), dtype=torch.float32, device=instr_logits_t.device)

    instr_prob = F.softmax(instr_logits_t, dim=-1)
    op_prob = F.softmax(op_logits_t, dim=-1)
    intraop_prob = F.softmax(intraop_logits_t, dim=-1)
    intracorp_prob = F.softmax(intracorp_logits_t, dim=-1)

    for j, gt in enumerate(gt_objects):
        gt_instr = int(gt["instrument"])
        gt_op = int(gt["operator"])
        gt_bbox = gt["tool_bbox"]
        gt_intraop = int(gt["intraoperative_track"])
        gt_intracorp = int(gt["intracorporeal_track"])

        for i in range(k):
            p_bbox = bbox_pred_t[i].detach().cpu().tolist()
            bbox_l1 = sum(abs(float(a) - float(b)) for a, b in zip(p_bbox, gt_bbox))
            instr_cost = -float(instr_prob[i, gt_instr].item())
            op_cost = -float(op_prob[i, gt_op].item())
            intraop_cost = -float(intraop_prob[i, gt_intraop].item())
            intracorp_cost = -float(intracorp_prob[i, gt_intracorp].item())

            total = (
                match_cost_bbox * bbox_l1
                + match_cost_instr * instr_cost
                + match_cost_op * op_cost
                + match_cost_intraop * intraop_cost
                + match_cost_intracorp * intracorp_cost
            )
            cost[i, j] = total

    return cost


def hungarian_match_one_t(
    instr_logits_t,
    op_logits_t,
    bbox_pred_t,
    intraop_logits_t,
    intracorp_logits_t,
    gt_objects
):
    k = instr_logits_t.shape[0]
    m = len(gt_objects)

    if m == 0:
        return [], list(range(k)), []

    cost = build_matching_cost_for_one_t(
        instr_logits_t,
        op_logits_t,
        bbox_pred_t,
        intraop_logits_t,
        intracorp_logits_t,
        gt_objects
    )

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

    matches = list(zip(row_ind.tolist(), col_ind.tolist()))
    matched_pred = set(r for r, _ in matches)
    unmatched_pred = [i for i in range(k) if i not in matched_pred]

    return matches, unmatched_pred, []


# -------------------------
# Loss
# -------------------------
def compute_taskb_loss(outputs, anchor_targets):
    phase_logits = outputs["phase_logits"][0]          # [T, P]
    obj_logits = outputs["obj_logits"][0]              # [T, K]
    instr_logits = outputs["instr_logits"][0]          # [T, K, C_instr]
    op_logits = outputs["op_logits"][0]                # [T, K, C_op]
    bbox_pred = outputs["bbox_pred"][0]                # [T, K, 4]
    intraop_logits = outputs["intraop_logits"][0]      # [T, K, C_intraop]
    intracorp_logits = outputs["intracorp_logits"][0]  # [T, K, C_intracorp]

    device = phase_logits.device
    t_len = len(anchor_targets)
    k = obj_logits.shape[1]

    # 1) phase loss
    phase_gt = torch.tensor(
        [int(x["phase"]) for x in anchor_targets],
        dtype=torch.long,
        device=device
    )
    loss_phase = F.cross_entropy(phase_logits, phase_gt)

    # 2) object/attr/bbox losses
    loss_obj = torch.tensor(0.0, device=device)
    loss_instr = torch.tensor(0.0, device=device)
    loss_op = torch.tensor(0.0, device=device)
    loss_bbox = torch.tensor(0.0, device=device)
    loss_intraop = torch.tensor(0.0, device=device)
    loss_intracorp = torch.tensor(0.0, device=device)

    count_obj_steps = 0
    count_matched = 0

    for t in range(t_len):
        gt_objects = anchor_targets[t]["objects"]

        matches, unmatched_pred, _ = hungarian_match_one_t(
            instr_logits[t],
            op_logits[t],
            bbox_pred[t],
            intraop_logits[t],
            intracorp_logits[t],
            gt_objects
        )

        # objectness GT
        obj_gt_t = torch.zeros((k,), dtype=torch.float32, device=device)
        for pred_idx, gt_idx in matches:
            obj_gt_t[pred_idx] = 1.0

        loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
            obj_logits[t], obj_gt_t
        )
        count_obj_steps += 1

        # matched pairs
        for pred_idx, gt_idx in matches:
            gt = gt_objects[gt_idx]

            gt_instr = torch.tensor([int(gt["instrument"])], dtype=torch.long, device=device)
            gt_op = torch.tensor([int(gt["operator"])], dtype=torch.long, device=device)
            gt_intraop = torch.tensor([int(gt["intraoperative_track"])], dtype=torch.long, device=device)
            gt_intracorp = torch.tensor([int(gt["intracorporeal_track"])], dtype=torch.long, device=device)
            gt_bbox = torch.tensor(gt["tool_bbox"], dtype=torch.float32, device=device)

            loss_instr = loss_instr + F.cross_entropy(
                instr_logits[t, pred_idx].unsqueeze(0), gt_instr
            )
            loss_op = loss_op + F.cross_entropy(
                op_logits[t, pred_idx].unsqueeze(0), gt_op
            )
            loss_intraop = loss_intraop + F.cross_entropy(
                intraop_logits[t, pred_idx].unsqueeze(0), gt_intraop
            )
            loss_intracorp = loss_intracorp + F.cross_entropy(
                intracorp_logits[t, pred_idx].unsqueeze(0), gt_intracorp
            )

            pred_box = bbox_pred[t, pred_idx]
            l1 = F.l1_loss(pred_box, gt_bbox)
            iou = bbox_iou_xywh_torch(pred_box, gt_bbox)
            bbox_loss = lambda_bbox_l1 * l1 + lambda_bbox_iou * (1.0 - iou)
            loss_bbox = loss_bbox + bbox_loss

            count_matched += 1

    if count_obj_steps > 0:
        loss_obj = loss_obj / count_obj_steps

    if count_matched > 0:
        loss_instr = loss_instr / count_matched
        loss_op = loss_op / count_matched
        loss_bbox = loss_bbox / count_matched
        loss_intraop = loss_intraop / count_matched
        loss_intracorp = loss_intracorp / count_matched
    else:
        loss_instr = torch.tensor(0.0, device=device)
        loss_op = torch.tensor(0.0, device=device)
        loss_bbox = torch.tensor(0.0, device=device)
        loss_intraop = torch.tensor(0.0, device=device)
        loss_intracorp = torch.tensor(0.0, device=device)

    total_loss = (
        lambda_phase * loss_phase
        + lambda_obj * loss_obj
        + lambda_instr * loss_instr
        + lambda_op * loss_op
        + loss_bbox
        + lambda_intraop * loss_intraop
        + lambda_intracorp * loss_intracorp
    )

    loss_dict = {
        "total": total_loss.detach().item(),
        "phase": loss_phase.detach().item(),
        "obj": loss_obj.detach().item(),
        "instr": loss_instr.detach().item(),
        "op": loss_op.detach().item(),
        "bbox": loss_bbox.detach().item(),
        "intraop": loss_intraop.detach().item(),
        "intracorp": loss_intracorp.detach().item(),
    }
    return total_loss, loss_dict


# -------------------------
# Train
# -------------------------
def train():
    train_rows = load_jsonl(train_jsonl)
    num_intraop_classes, num_intracorp_classes = infer_track_vocab_sizes(train_rows)

    print("num_intraop_classes:", num_intraop_classes)
    print("num_intracorp_classes:", num_intracorp_classes)

    train_dataset = TaskBDataset(train_jsonl, max_frames_per_video=max_frames_per_video)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = TaskBQwen25VLLoRA(
        model_name=model_name,
        num_phase_classes=num_phase_classes,
        num_instrument_classes=num_instrument_classes,
        num_operator_classes=num_operator_classes,
        num_intraop_classes=num_intraop_classes,
        num_intracorp_classes=num_intracorp_classes,
        num_slots=num_slots,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, eps=adam_eps)

    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        running_total = 0.0
        steps = 0
        avg_phase = 0.0
        avg_obj = 0.0
        avg_instr = 0.0
        avg_op = 0.0
        avg_bbox = 0.0
        avg_intraop = 0.0
        avg_intracorp = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for sample in pbar:
            global_step += 1

            try:
                outputs = model(sample["frames"])
            except Exception as e:
                print("\n[FORWARD FAIL]")
                print("video:", sample["video_path"])
                print("frame_indices:", sample["frame_indices"])
                raise e

            loss, loss_dict = compute_taskb_loss(outputs, sample["anchor_targets"])

            if torch.isnan(loss) or torch.isinf(loss):
                print("\n[BAD LOSS]")
                print("video:", sample["video_path"])
                print("frame_indices:", sample["frame_indices"])
                raise ValueError("loss invalid")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            optimizer.step()

            running_total += loss.item()
            avg_phase += loss_dict["phase"]
            avg_obj += loss_dict["obj"]
            avg_instr += loss_dict["instr"]
            avg_op += loss_dict["op"]
            avg_bbox += loss_dict["bbox"]
            avg_intraop += loss_dict["intraop"]
            avg_intracorp += loss_dict["intracorp"]
            steps += 1

            pbar.set_postfix(
                total=f"{running_total/max(steps,1):.4f}",
                phase=f"{avg_phase/max(steps,1):.4f}",
                obj=f"{avg_obj/max(steps,1):.4f}",
                bbox=f"{avg_bbox/max(steps,1):.4f}",
                step=global_step
            )

        epoch_total = running_total / max(steps, 1)
        print(
            f"\nEpoch {epoch+1}/{num_epochs} done | "
            f"total={epoch_total:.4f} | "
            f"phase={avg_phase/max(steps,1):.4f} | "
            f"obj={avg_obj/max(steps,1):.4f} | "
            f"instr={avg_instr/max(steps,1):.4f} | "
            f"op={avg_op/max(steps,1):.4f} | "
            f"bbox={avg_bbox/max(steps,1):.4f} | "
            f"intraop={avg_intraop/max(steps,1):.4f} | "
            f"intracorp={avg_intracorp/max(steps,1):.4f}\n"
        )

    ckpt = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch_total_loss": epoch_total,
        "num_intraop_classes": num_intraop_classes,
        "num_intracorp_classes": num_intracorp_classes,
        "num_slots": num_slots,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }

    #torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))
    torch.save(ckpt, os.path.join(save_dir, "last.pt"))

    print("Training done.")
    print("Saved to:", save_dir)


if __name__ == "__main__":
    train()