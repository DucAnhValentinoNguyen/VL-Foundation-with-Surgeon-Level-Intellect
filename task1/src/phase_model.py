# Phase recognition model based on Qwen2.5-VL.
# It supports two backbone modes:
# 1) frozen backbone, where only the temporal module and classification head are trained;
# 2) LoRA adaptation, where parameter-efficient adapters are added to the backbone.
# For each sampled frame, the model extracts a visual-text feature, then applies
# a BiLSTM and a linear head to predict frame-wise phase labels.

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLModel
from peft import LoraConfig, get_peft_model

class TaskAQwen25VL(nn.Module):
    def __init__(
        self,
        model_name,
        num_phase_classes=7,
        use_lora=False,
        freeze_backbone=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        backbone_dtype=None,
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.use_lora = use_lora
        self.freeze_backbone = freeze_backbone

        if backbone_dtype is None:
            backbone_dtype = torch.float32 if freeze_backbone else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

        backbone = Qwen2_5_VLModel.from_pretrained(
            model_name,
            torch_dtype=backbone_dtype
        )

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            backbone = get_peft_model(backbone, lora_config)
            backbone.print_trainable_parameters()

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()

        self.backbone = backbone

        hidden_size = self.backbone.config.text_config.hidden_size

        self.feat_norm = nn.LayerNorm(hidden_size)
        self.temporal = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.phase_head = nn.Linear(hidden_size, num_phase_classes)

    def encode_one_frame(self, frame_rgb):
        device = next(self.phase_head.parameters()).device

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

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**inputs, return_dict=True)
        else:
            outputs = self.backbone(**inputs, return_dict=True)

        hidden = outputs.last_hidden_state.float()
        feat = hidden.mean(dim=1)[0]

        if torch.isnan(feat).any() or torch.isinf(feat).any():
            raise ValueError("feature has NaN/Inf inside encode_one_frame")

        return feat

    def forward(self, frames):
        feats = []
        for frame_rgb in frames:
            feats.append(self.encode_one_frame(frame_rgb))

        feats = torch.stack(feats, dim=0).unsqueeze(0)   # [1, T, D]
        feats = self.feat_norm(feats)

        if torch.isnan(feats).any() or torch.isinf(feats).any():
            raise ValueError("feats has NaN/Inf after feat_norm")

        temporal_out, _ = self.temporal(feats)
        phase_logits = self.phase_head(temporal_out)

        if torch.isnan(phase_logits).any() or torch.isinf(phase_logits).any():
            raise ValueError("phase_logits has NaN/Inf")

        return phase_logits