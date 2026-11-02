"""Excision model: bidirectional Qwen3 backbone with a token head and a window head (report Section 3.2).

Input: token ids and attention mask of a batch of windows. Output: tag logits [N, T, 5] (O, B, I, E, S) and one
window logit [N] from the masked mean of the token states.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3Model

from excise.config import BACKBONE


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ExcisionModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.token_head = nn.Linear(backbone.config.hidden_size, 5)
        self.window_head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        return self.token_head(h), self.window_head(pooled).squeeze(-1)

    @classmethod
    def from_backbone(cls, path=BACKBONE["repo"]):
        """Pretrained backbone with causal masking switched off, fresh heads, fp32 for training."""
        revision = BACKBONE["revision"] if path == BACKBONE["repo"] else None
        cfg = Qwen3Config.from_pretrained(path, revision=revision)
        cfg.is_causal = False
        backbone = Qwen3Model.from_pretrained(
            path, revision=revision, config=cfg, dtype=torch.float32, attn_implementation="sdpa"
        )
        return cls(backbone)

    @classmethod
    def from_dir(cls, path):
        """Trained weights saved by `save`: bf16 backbone, fp32 heads."""
        cfg = Qwen3Config.from_pretrained(path)
        cfg.is_causal = False
        model = cls(Qwen3Model(cfg))
        model.load_state_dict(load_file(f"{path}/model.safetensors"))
        model.backbone.to(torch.bfloat16)
        return model

    def save(self, path):
        state = {
            k: (v.to(torch.bfloat16) if k.startswith("backbone.") else v.float()).contiguous()
            for k, v in self.state_dict().items()
        }
        save_file(state, f"{path}/model.safetensors")
        self.backbone.config.save_pretrained(path)
