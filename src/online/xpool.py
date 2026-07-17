from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class XPoolMultiHeadedAttention(nn.Module):
    """Cross-modal text-to-video-frame attention from the official X-Pool code."""

    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        text_embeds: torch.Tensor,
        video_embeds: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeds: [num_texts, embed_dim]
            video_embeds: [num_videos, num_frames, embed_dim]

        Returns:
            pooled video embeddings conditioned on each text: [num_videos, num_texts, embed_dim]
        """
        num_texts, _ = text_embeds.shape
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0)

        num_videos, num_frames, _ = video_embeds.shape
        k = self.k_proj(video_embeds)
        k = k.reshape(num_videos, num_frames, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = self.v_proj(video_embeds)
        v = v.reshape(num_videos, num_frames, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 3, 1)

        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        attention = v @ attention_weights
        attention = attention.permute(0, 3, 1, 2)
        attention = attention.reshape(num_videos, num_texts, self.embed_dim)
        output = self.out_proj(attention)
        if return_attention:
            return output, attention_weights
        return output


class XPoolTransformer(nn.Module):
    """X-Pool transformer pooling block, matching layer order and initialization."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cross_attn = XPoolMultiHeadedAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(dropout))
        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "linear" in name or "proj" in name:
                if "weight" in name:
                    nn.init.eye_(param)
                elif "bias" in name:
                    param.data.fill_(0.0)

    def forward(
        self,
        text_embeds: torch.Tensor,
        video_embeds: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)
        if return_attention:
            attn_out, attention_weights = self.cross_attn(text_embeds, video_embeds, return_attention=True)
        else:
            attn_out = self.cross_attn(text_embeds, video_embeds)
            attention_weights = None
        attn_out = self.layer_norm2(attn_out)
        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)
        if return_attention:
            assert attention_weights is not None
            return out, attention_weights
        return out


def load_xpool_transformer_checkpoint(module: XPoolTransformer, checkpoint_path: str | None) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"checkpoint_path": None, "loaded_keys": 0, "skipped_keys": 0}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"X-Pool checkpoint not found: {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    module_state = module.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state_dict.items():
        stripped_key = str(key)
        if stripped_key.startswith("module."):
            stripped_key = stripped_key[len("module.") :]
        if stripped_key.startswith("pool_frames."):
            stripped_key = stripped_key[len("pool_frames.") :]
        elif stripped_key.startswith("model.pool_frames."):
            stripped_key = stripped_key[len("model.pool_frames.") :]
        else:
            skipped += 1
            continue
        if stripped_key in module_state and tuple(module_state[stripped_key].shape) == tuple(value.shape):
            compatible[stripped_key] = value
        else:
            skipped += 1

    module.load_state_dict(compatible, strict=False)
    return {
        "checkpoint_path": str(path),
        "loaded_keys": len(compatible),
        "skipped_keys": skipped,
    }
