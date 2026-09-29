"""
Raksam Brain V3 (RB3)

Created by: Prosanto Raksam

From-scratch Transformer language model.
No external AI runtime is required.

Architecture:
- RMSNorm
- RoPE
- Grouped Query Attention (GQA)
- SwiGLU feed-forward network
- Residual scaling
- Weight tying
- Causal attention
- Safe autoregressive generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BrainConfig:
    vocab_size: int

    block_size: int = 256

    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2

    n_embd: int = 256
    ffn_mult: float = 2.6667

    dropout: float = 0.05
    rope_theta: float = 10000.0

    tie_embeddings: bool = True
    residual_scale: bool = True


# ============================================================
# RMS NORM
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x


# ============================================================
# ROTARY POSITION EMBEDDINGS
# ============================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the final dimension by 90 degrees in pairs.
    """

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Apply rotary positional embeddings to Q and K.

    q:
        [B, H, T, D]

    k:
        [B, Hkv, T, D]

    cos/sin:
        [1, 1, T, D]
    """

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        device = self.inv_freq.device

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(
            positions,
            self.inv_freq,
        )

        emb = torch.cat(
            [freqs, freqs],
            dim=-1,
        )

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        self.register_buffer(
            "cos_cached",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin,
            persistent=False,
        )

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ):
        if seq_len > self.cos_cached.shape[-2]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)

        return cos, sin


# ============================================================
# GROUPED QUERY ATTENTION
# ============================================================

class GQAttention(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()

        if config.n_head % config.n_kv_head != 0:
            raise ValueError(
                "n_head must be divisible by n_kv_head."
            )

        if config.n_embd % config.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head."
            )

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head

        self.head_dim = config.n_embd // config.n_head

        self.group_size = (
            config.n_head // config.n_kv_head
        )

        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.block_size,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_kv_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_kv_head,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = self.rope(
            T,
            device=x.device,
        )

        q, k = apply_rope(
            q,
            k,
            cos,
            sin,
        )

        # ----------------------------------------------------
        # Repeat KV heads for grouped-query attention.
        # ----------------------------------------------------

        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(
                self.group_size,
                dim=1,
            )

            v = v.repeat_interleave(
                self.group_size,
                dim=1,
            )

        # ----------------------------------------------------
        # Causal self-attention.
        # ----------------------------------------------------

        is_causal = attention_mask is None

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=(
                self.attn_dropout
                if self.training
                else 0.0
            ),
            is_causal=is_causal,
        )

        y = y.transpose(
            1,
            2,
        ).contiguous()

        y = y.view(
            B,
            T,
            C,
        )

        y = self.out_proj(y)

        y = self.resid_dropout(y)

        return y


# ============================================================
# SWIGLU
# ============================================================

class SwiGLU(nn.Module):
    def __init__(
        self,
        config: BrainConfig,
    ):
        super().__init__()

        hidden_dim = int(
            config.n_embd * config.ffn_mult
        )

        # Make hidden dimension friendly to 8.
        hidden_dim = max(
            8,
            ((hidden_dim + 7) // 8) * 8,
        )

        self.gate_proj = nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=False,
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=False,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        x = gate * up

        x = self.down_proj(x)

        x = self.dropout(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):
    def __init__(
        self,
        config: BrainConfig,
    ):
        super().__init__()

        self.norm1 = RMSNorm(
            config.n_embd
        )

        self.attn = GQAttention(
            config
        )

        self.norm2 = RMSNorm(
            config.n_embd
        )

        self.ffn = SwiGLU(
            config
        )

        if config.residual_scale:
            scale = 1.0 / math.sqrt(
                2.0 * config.n_layer
            )
        else:
            scale = 1.0

        self.residual_scale = scale

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x + (
            self.attn(
                self.norm1(x)
            )
            * self.residual_scale
        )

        x = x + (
            self.ffn(
                self.norm2(x)
            )
            * self.residual_scale
        )

        return x


# ============================================================
# RAKSAM BRAIN MODEL
# ============================================================

class BrainModel(nn.Module):
    def __init__(
        self,
        config: BrainConfig,
    ):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList(
            [
                Block(config)
                for _ in range(config.n_layer)
            ]
        )

        self.norm = RMSNorm(
            config.n_embd
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Weight tying.
        # ----------------------------------------------------

        if config.tie_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

        self.apply(self._init_weights)

        # Slightly smaller initialization for
        # residual projections.
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(
                        config.n_layer
                    ),
                )

            elif name.endswith("down_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(
                        config.n_layer
                    ),
                )

        print(
            f"Raksam Brain v3 initialized: "
            f"{self.num_parameters() / 1e6:.2f}M parameters"
        )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    @staticmethod
    def _init_weights(
        module: nn.Module,
    ):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    def num_parameters(
        self,
    ) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block size {self.config.block_size}."
            )

        x = self.token_embedding(idx)

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1),
                ),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ========================================================
    # SAFE SAMPLING
    # ========================================================

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        idx: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:

        if penalty <= 1.0:
            return logits

        # Apply penalty only to tokens already seen.
        seen_tokens = torch.unique(
            idx,
            dim=1,
        )

        seen_tokens = seen_tokens[
            seen_tokens < logits.size(-1)
        ]

        if seen_tokens.numel() == 0:
            return logits

        for token_id in seen_tokens:
            token_id = int(token_id)

            value = logits[
                :,
                -1,
                token_id,
            ]

            if value >= 0:
                logits[
                    :,
                    -1,
                    token_id,
                ] = value / penalty
            else:
                logits[
                    :,
                    -1,
                    token_id,
                ] = value * penalty

        return logits

    @staticmethod
    def _top_k_filter(
        logits: torch.Tensor,
        top_k: int | None,
    ) -> torch.Tensor:

        if (
            top_k is None
            or top_k <= 0
            or top_k >= logits.size(-1)
        ):
            return logits

        values, _ = torch.topk(
            logits,
            top_k,
            dim=-1,
        )

        threshold = values[
            ...,
            -1,
            None,
        ]

        logits = logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

        return logits

    @staticmethod
    def _top_p_filter(
        logits: torch.Tensor,
        top_p: float | None,
    ) -> torch.Tensor:

        if (
            top_p is None
            or top_p >= 1.0
            or top_p <= 0.0
        ):
            return logits

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probs = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        remove = cumulative_probs > top_p

        # Always keep the highest-probability token.
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf"),
        )

        logits = torch.full_like(
            logits,
            float("-inf"),
        )

        logits.scatter_(
            -1,
            sorted_indices,
            sorted_logits,
        )

        return logits

    # ========================================================
    # GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_ids: list[int] | None = None,

        # ----------------------------------------------------
        # NEW:
        # Minimum number of tokens generated before EOS
        # is allowed.
        #
        # This prevents the under-trained model from doing:
        #
        #     "What is your name?"
        #             ↓
        #             "{"
        #             ↓
        #            <EOS>
        #
        # ----------------------------------------------------
        min_new_tokens: int = 0,
    ) -> torch.Tensor:

        if idx.ndim != 2:
            raise ValueError(
                "idx must have shape [batch, sequence]."
            )

        stop_set = set(
            int(x)
            for x in (stop_ids or [])
        )

        max_new_tokens = max(
            0,
            int(max_new_tokens),
        )

        min_new_tokens = max(
            0,
            int(min_new_tokens),
        )

        # Never require more minimum tokens than
        # we are actually going to generate.
        min_new_tokens = min(
            min_new_tokens,
            max_new_tokens,
        )

        temperature = max(
            float(temperature),
            1e-4,
        )

        for step in range(
            max_new_tokens
        ):

            # ------------------------------------------------
            # Keep context inside model block size.
            # ------------------------------------------------

            idx_cond = idx[
                :,
                -self.config.block_size:,
            ]

            logits, _ = self(
                idx_cond,
                None,
            )

            logits = logits[
                :,
                -1,
                :,
            ]

            # ------------------------------------------------
            # Temperature.
            # ------------------------------------------------

            logits = logits / temperature

            # ------------------------------------------------
            # Repetition penalty.
            # ------------------------------------------------

            logits = self._apply_repetition_penalty(
                logits.unsqueeze(1),
                idx_cond,
                repetition_penalty,
            ).squeeze(1)

            # ------------------------------------------------
            # IMPORTANT FIX:
            #
            # Do NOT allow EOS during the first
            # min_new_tokens generation steps.
            #
            # This is especially useful for a small,
            # still-learning model.
            # ------------------------------------------------

            if (
                stop_set
                and step < min_new_tokens
            ):
                for stop_id in stop_set:

                    if (
                        0 <= stop_id
                        < logits.size(-1)
                    ):
                        logits[
                            :,
                            stop_id,
                        ] = float("-inf")

            # ------------------------------------------------
            # Remove NaN / infinity problems.
            # ------------------------------------------------

            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )

            # ------------------------------------------------
            # Top-K.
            # ------------------------------------------------

            logits = self._top_k_filter(
                logits,
                top_k,
            )

            # ------------------------------------------------
            # Top-P.
            # ------------------------------------------------

            logits = self._top_p_filter(
                logits,
                top_p,
            )

            # ------------------------------------------------
            # Safety fallback.
            #
            # If filtering accidentally removes everything,
            # restore the original highest-probability token.
            # ------------------------------------------------

            if not torch.isfinite(
                logits
            ).any():

                logits = torch.nan_to_num(
                    logits,
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

            probs = F.softmax(
                logits,
                dim=-1,
            )

            # ------------------------------------------------
            # Another numerical safety fallback.
            # ------------------------------------------------

            if (
                not torch.isfinite(probs).all()
                or probs.sum(dim=-1).min() <= 0
            ):
                probs = torch.zeros_like(
                    logits
                )

                best_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

                probs.scatter_(
                    -1,
                    best_token,
                    1.0,
                )

            else:
                probs = probs / probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)

            # ------------------------------------------------
            # Sample next token.
            # ------------------------------------------------

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                [
                    idx,
                    next_token,
                ],
                dim=1,
            )

            # ------------------------------------------------
            # EOS is now allowed after min_new_tokens.
            # ------------------------------------------------

            if stop_set:

                token_value = int(
                    next_token[0, 0]
                )

                if (
                    step >= min_new_tokens
                    and token_value in stop_set
                ):
                    break

        return idx