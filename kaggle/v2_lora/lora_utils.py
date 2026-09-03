"""Minimal auditable LoRA for open_clip ViT visual transformer (last N blocks)."""

from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.linear = linear
        self.scale = alpha / max(1, r)
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.drop = nn.Dropout(dropout)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


class LoRAInProjDelta(nn.Module):
    def __init__(self, embed_dim: int, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.scale = alpha / max(1, r)
        self.lora_A = nn.Parameter(torch.zeros(r, embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3 * embed_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


def _lora_attention_forward(block, x: torch.Tensor) -> torch.Tensor:
    attn = block.attn
    in_delta: LoRAInProjDelta = block.lora_in_delta
    out_lora: LoRALinear = block.lora_out_proj
    embed_dim = attn.embed_dim
    num_heads = attn.num_heads
    tgt_len, bsz, _ = x.shape
    flat = x.reshape(tgt_len * bsz, embed_dim)
    qkv = F.linear(flat, attn.in_proj_weight, attn.in_proj_bias) + in_delta(flat)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(tgt_len, bsz, embed_dim).transpose(0, 1)
    k = k.view(tgt_len, bsz, embed_dim).transpose(0, 1)
    v = v.view(tgt_len, bsz, embed_dim).transpose(0, 1)
    attn_out, _ = F.multi_head_attention_forward(
        query=q,
        key=k,
        value=v,
        embed_dim_to_check=embed_dim,
        num_heads=num_heads,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=attn.bias_k,
        bias_v=attn.bias_v,
        add_zero_attn=attn.add_zero_attn,
        dropout_p=attn.dropout if attn.training else 0.0,
        out_proj_weight=out_lora.linear.weight,
        out_proj_bias=out_lora.linear.bias,
        training=attn.training,
        need_weights=False,
    )
    flat_out = attn_out.transpose(0, 1).reshape(tgt_len * bsz, embed_dim)
    lora = out_lora.drop(flat_out) @ out_lora.lora_A.T @ out_lora.lora_B.T * out_lora.scale
    out = (flat_out + lora).view(tgt_len, bsz, embed_dim)
    return out


def inject_lora_last_blocks(visual: nn.Module, last_n: int = 4, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> list[str]:
    blocks = visual.transformer.resblocks
    touched = []
    n = len(blocks)
    for i in range(n - last_n, n):
        blk = blocks[i]
        attn = blk.attn
        for p in attn.parameters():
            p.requires_grad = False
        blk.lora_in_delta = LoRAInProjDelta(attn.embed_dim, r, alpha, dropout)
        blk.lora_out_proj = LoRALinear(attn.out_proj, r, alpha, dropout)
        blk.attention = MethodType(lambda self, x, _blk=blk: _lora_attention_forward(_blk, x), blk)
        touched.append(f"visual.transformer.resblocks.{i}.attn")
    return touched


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_pct": float(100.0 * trainable / max(1, total)),
    }
