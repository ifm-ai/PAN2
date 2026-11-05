import math
from typing import Any, List, Tuple, Optional, Union, Dict

from dataclasses import dataclass, fields
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import loguru

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


from .modules.layers import RMSNorm, AdaLNProjection, activation_factory, modulate, gate
from .modules.embeddings import TimestepEncoder, VideoPatchEmbed
from .modules.attention import joint_attention
from pan.utils.flash_attention import flash_attn_varlen
from pan.utils.ulysses import ulysses_flash_attn_varlen
from .modules.rope import apply_rope, rope_tables_for_grid
from .modules.output_head import OutputHead
from .modules.caption_encoder import CaptionEncoder
from .modules.triton_kernels import (
    packed_simple_concat,
    packed_gate,
    packed_modulate,
    packed_concat_mse_loss,
)
from .modules.mlp_ops import grouped_out_proj, grouped_qkv_proj, grouped_ffn

from pan.commons.parallel_layout import current_parallel_layout


class GroupedQKVProj(nn.Module):
    """Merged video+text QKV projection with per-modality RMS q/k norm.

    Stores raw parameters (not ``nn.Linear`` / ``RMSNorm`` submodules) so a later
    quack grouped-matmul kernel can consume them directly. Video and text weights
    are stacked along dim 0 (3 projections per modality -- each modality block
    is [Q | K | V])::

        weight:   (2 * hidden_size * 3, hidden_size)  -> view(2, 3*hidden, hidden)
        bias:     (2 * hidden_size * 3,)              -> view(2, 3*hidden)
        q_weight / k_weight: (2 * head_dim,)          -> (head_dim,) per modality,
            the head_dim-shared RMS affine (broadcast over heads), matching the
            old per-modality RMSNorm(head_dim) exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        eps: float = 1e-6,
    ):
        super().__init__()
        assert qk_norm, "GroupedQKVProj requires qk_norm=True."
        assert qk_norm_type == "rms", "GroupedQKVProj only supports rms qk norm."
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(2 * hidden_size * 3, hidden_size))
        self.bias = nn.Parameter(torch.empty(2 * hidden_size * 3))
        self.q_weight = nn.Parameter(torch.ones(2 * self.head_dim))
        self.k_weight = nn.Parameter(torch.ones(2 * self.head_dim))
        # Muon treats each (head_dim, hidden) block as a chunk:
        # 3 (q,k,v) x 2 modalities x heads.
        self.weight._muon_chunks_dim_0 = num_heads * 2 * 3

        self.reset_parameters()

    def reset_parameters(self):
        # Every (modality, q/k/v) block matches nn.Linear's default init (fan_in = hidden_size).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: (..., num_heads, head_dim); weight: (head_dim,) broadcast over heads.
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out.type_as(x)
        return out * weight

    def _project(self, x, w, b, qw, kw):
        q, k, v = rearrange(
            F.linear(x, w, b), "... (K H D) -> K ... H D", K=3, H=self.num_heads
        ).unbind(dim=0)
        q = self._rms(q, qw).to(v)
        k = self._rms(k, kw).to(v)
        return q, k, v

    def project(self, x: torch.Tensor, modality: str):
        # Per-modality block is [Q | K | V]: video rows first, then text rows.
        h3, d = self.hidden_size * 3, self.head_dim
        if modality == "video":
            return self._project(x, self.weight[:h3], self.bias[:h3], self.q_weight[:d], self.k_weight[:d])
        if modality == "text":
            return self._project(
                x, self.weight[h3:2 * h3], self.bias[h3:2 * h3], self.q_weight[d:], self.k_weight[d:]
            )
        raise ValueError(f"Unknown modality: {modality}")

    def forward(
        self,
        x_mod: torch.Tensor,
        cu_seqlens_m: torch.Tensor,
        indices_interleave_to_concat: torch.Tensor,
        indices_concat_to_interleave: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ):
        """Fused packed-path projection: linear (video/text via cu_seqlens_m) ->
        per-head RMSNorm (Q/K only) -> permute (concat<->interleave) -> RoPE
        (Q/K only), as one autograd op (see grouped_qkv_proj). Returns q, k, v
        already in interleave layout, shape [total_seqlen, num_heads, head_dim],
        ready for flash attention -- no separate index_select/apply_rope
        needed at the call site."""
        cos, sin = freqs_cis
        h3 = 3 * self.hidden_size
        weight = self.weight.view(2, h3, self.hidden_size)
        bias = self.bias.view(2, h3)
        return grouped_qkv_proj(
            x_mod, weight, bias, self.q_weight, self.k_weight,
            indices_interleave_to_concat, indices_concat_to_interleave,
            cu_seqlens_m, cos, sin, self.num_heads, self.head_dim, self.eps,
        )


class GroupedOutProj(nn.Module):
    """Merged video+text attention output projection.

    weight: (2 * hidden_size, hidden_size) -> view(2, hidden, hidden), one
    (hidden, hidden) block per modality. bias: (2 * hidden_size,).
    """

    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(2 * hidden_size))
        # One (hidden, hidden) matrix per modality.
        self.weight._muon_chunks_dim_0 = 2
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.bias, -bound, bound)

    def project(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.hidden_size
        if modality == "video":
            return F.linear(x, self.weight[:h], self.bias[:h])
        if modality == "text":
            return F.linear(x, self.weight[h:], self.bias[h:])
        raise ValueError(f"Unknown modality: {modality}")

    def forward(self, attn: torch.Tensor, gather_idx: torch.Tensor, cu_seqlens_m: torch.Tensor) -> torch.Tensor:
        """Fused packed-path projection: gathers `attn` (interleave layout) via
        `gather_idx` (the full concat<->interleave permutation) straight into the
        concat layout while applying the per-modality (weight, bias) selected by
        `cu_seqlens_m` = [0, total_video, total_seqlen]. See grouped_out_proj."""
        h = self.hidden_size
        weight = self.weight.view(2, h, h)
        bias = self.bias.view(2, h)
        return grouped_out_proj(attn, weight, bias, gather_idx, cu_seqlens_m)


class GroupedFFN(nn.Module):
    """Per-modality GELU(tanh) MLP storing raw weight/bias parameters.

    Video and text FFNs are kept as separate parameters (their hidden dims can
    differ, e.g. 8x vs 4x width) so a later quack grouped-matmul kernel can treat
    them as two groups with distinct problem sizes.
    """

    def __init__(
        self,
        hidden_size: int,
        video_mlp_hidden_dim: int,
        text_mlp_hidden_dim: int,
        mlp_act_type: str = "gelu_tanh",
    ):
        super().__init__()
        assert mlp_act_type == "gelu_tanh", "GroupedFFN only supports the gelu_tanh MLP."
        # grouped_ffn's fused up-projection buffer is allocated video_hidden-wide
        # and text writes only its first text_hidden columns -- video must be at
        # least as wide as text (equality is fine, they need not match).
        assert video_mlp_hidden_dim >= text_mlp_hidden_dim, (
            "GroupedFFN requires video_mlp_hidden_dim >= text_mlp_hidden_dim."
        )
        self.hidden_size = hidden_size
        self.video_mlp_hidden_dim = video_mlp_hidden_dim
        self.text_mlp_hidden_dim = text_mlp_hidden_dim

        self.w_up_video = nn.Parameter(torch.empty(video_mlp_hidden_dim, hidden_size))
        self.b_up_video = nn.Parameter(torch.empty(video_mlp_hidden_dim))
        self.w_down_video = nn.Parameter(torch.empty(hidden_size, video_mlp_hidden_dim))
        self.b_down_video = nn.Parameter(torch.empty(hidden_size))

        self.w_up_text = nn.Parameter(torch.empty(text_mlp_hidden_dim, hidden_size))
        self.b_up_text = nn.Parameter(torch.empty(text_mlp_hidden_dim))
        self.w_down_text = nn.Parameter(torch.empty(hidden_size, text_mlp_hidden_dim))
        self.b_down_text = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def _reset_linear(self, weight: torch.Tensor, bias: torch.Tensor):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(weight.shape[1])
        nn.init.uniform_(bias, -bound, bound)

    def reset_parameters(self):
        self._reset_linear(self.w_up_video, self.b_up_video)
        self._reset_linear(self.w_down_video, self.b_down_video)
        self._reset_linear(self.w_up_text, self.b_up_text)
        self._reset_linear(self.w_down_text, self.b_down_text)

    def project(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "video":
            up = F.gelu(F.linear(x, self.w_up_video, self.b_up_video), approximate="tanh")
            return F.linear(up, self.w_down_video, self.b_down_video)
        if modality == "text":
            up = F.gelu(F.linear(x, self.w_up_text, self.b_up_text), approximate="tanh")
            return F.linear(up, self.w_down_text, self.b_down_text)
        raise ValueError(f"Unknown modality: {modality}")

    def forward(self, x: torch.Tensor, cu_seqlens_m: torch.Tensor) -> torch.Tensor:
        """Fused packed-path FFN: one quack gemm_act + one quack gemm per
        modality (see grouped_ffn), with video/text row ranges selected from
        `cu_seqlens_m` = [0, total_video, total_active] entirely on-device (no
        .item() split). Padding rows (beyond total_active) are left
        uninitialized -- nothing downstream reads them."""
        return grouped_ffn(
            x,
            self.w_up_video, self.b_up_video, self.w_down_video, self.b_down_video,
            self.w_up_text, self.b_up_text, self.w_down_text, self.b_down_text,
            cu_seqlens_m,
        )


class JointAttentionBlock(nn.Module):
    """Joint-attention block with dense modality-specific projections and FFNs.

    Video and text projections are fused into ``GroupedQKVProj`` / ``GroupedOutProj`` / ``GroupedFFN`` modules
    that hold raw parameters for later quack grouped-matmul kernels.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attention_backend: Optional[str] = None,
        normalize_qk: bool = True,
        qk_norm_kind: str = "rms",
        ffn_activation: str = "gelu_tanh",
        mlp_ratio: float = 8.0,
        context_mlp_ratio: float = 4.0,
        zero_init_modulation: bool = True,
        text_modulation_timestep_dependent: bool = False,
    ):
        super().__init__()
        self.attention_backend = attention_backend
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.text_mlp_hidden_dim = int(hidden_size * context_mlp_ratio)

        # Fused video+text QKV projection (+ per-modality rms q/k norm) and output
        # projection: raw parameters merged along dim 0 for later quack kernels.
        # Creation order below fixes the RNG stream of the initializers; keep it.
        self.qkv_proj = GroupedQKVProj(hidden_size, num_heads, qk_norm=normalize_qk, qk_norm_type=qk_norm_kind)
        self.out_proj = GroupedOutProj(hidden_size)

        # pre_norm / norm2 are affine-free LayerNorms shared across video and text
        # (the two modalities' norms were identical, so a single instance suffices).
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        silu = activation_factory("silu")
        # shift/scale/gate for attention and FFN, video tokens: 6 * hidden from the timestep vector.
        self.modulation = AdaLNProjection(hidden_size, 6, silu, zero_init=zero_init_modulation)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = GroupedFFN(hidden_size, self.mlp_hidden_dim, self.text_mlp_hidden_dim, mlp_act_type=ffn_activation)

        self.text_modulation_timestep_dependent = text_modulation_timestep_dependent
        if text_modulation_timestep_dependent:
            self.text_modulation = AdaLNProjection(hidden_size, 6, silu, zero_init=zero_init_modulation)
        else:
            # Text modulation is sample-independent, so learn the final 6-way modulation vector directly.
            self.text_modulation = nn.Parameter(torch.empty(hidden_size * 6))
            if zero_init_modulation:
                nn.init.zeros_(self.text_modulation)
            else:
                nn.init.uniform_(self.text_modulation, -hidden_size ** -0.5, hidden_size ** -0.5)


    def _text_modulation(
        self,
        batch_size: int,
        ref: torch.Tensor,
        vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.text_modulation_timestep_dependent:
            if vec is None:
                raise ValueError("Timestep-dependent text modulation requires vec.")
            return self.text_modulation(vec).to(dtype=ref.dtype)
        return self.text_modulation.unsqueeze(0).expand(batch_size, -1).to(dtype=ref.dtype)

    def _convert_grouped_state_dict(self, state_dict, prefix):
        """Convert a previous-format checkpoint (separate video/text
        ``linear1_qkv`` / ``q_norm`` / ``attn_out`` / ``ffn`` modules) into the
        merged ``GroupedQKVProj`` / ``GroupedOutProj`` / ``GroupedFFN`` layout.
        No-op when the checkpoint is already in the new format."""
        P = prefix

        # QKV: stack video then text along dim 0 (each block is [Q|K|V] all heads).
        if P + "qkv_proj.weight" not in state_dict and P + "linear1_qkv.weight" in state_dict:
            vw = state_dict.pop(P + "linear1_qkv.weight")
            tw = state_dict.pop(P + "text_linear1_qkv.weight")
            state_dict[P + "qkv_proj.weight"] = torch.cat([vw, tw], dim=0)
            vb = state_dict.pop(P + "linear1_qkv.bias")
            tb = state_dict.pop(P + "text_linear1_qkv.bias")
            state_dict[P + "qkv_proj.bias"] = torch.cat([vb, tb], dim=0)

        # q/k norm: old affine is head_dim-shared per modality; stack video then
        # text -> (2 * head_dim,).
        if P + "qkv_proj.q_weight" not in state_dict and P + "q_norm.weight" in state_dict:
            vq = state_dict.pop(P + "q_norm.weight")
            tq = state_dict.pop(P + "text_q_norm.weight")
            state_dict[P + "qkv_proj.q_weight"] = torch.cat([vq, tq], dim=0)
            vk = state_dict.pop(P + "k_norm.weight")
            tk = state_dict.pop(P + "text_k_norm.weight")
            state_dict[P + "qkv_proj.k_weight"] = torch.cat([vk, tk], dim=0)

        # Output projection: stack video then text -> (2 * hidden, hidden).
        if P + "out_proj.weight" not in state_dict and P + "attn_out.weight" in state_dict:
            vw = state_dict.pop(P + "attn_out.weight")
            tw = state_dict.pop(P + "text_attn_out.weight")
            state_dict[P + "out_proj.weight"] = torch.cat([vw, tw], dim=0)
            vb = state_dict.pop(P + "attn_out.bias")
            tb = state_dict.pop(P + "text_attn_out.bias")
            state_dict[P + "out_proj.bias"] = torch.cat([vb, tb], dim=0)

        # FFN: kept per-modality; direct key rename (gelu MLP: .0 = up, .2 = down).
        if P + "ffn.w_up_video" not in state_dict and P + "ffn.0.weight" in state_dict:
            state_dict[P + "ffn.w_up_video"] = state_dict.pop(P + "ffn.0.weight")
            state_dict[P + "ffn.b_up_video"] = state_dict.pop(P + "ffn.0.bias")
            state_dict[P + "ffn.w_down_video"] = state_dict.pop(P + "ffn.2.weight")
            state_dict[P + "ffn.b_down_video"] = state_dict.pop(P + "ffn.2.bias")
            state_dict[P + "ffn.w_up_text"] = state_dict.pop(P + "text_ffn.0.weight")
            state_dict[P + "ffn.b_up_text"] = state_dict.pop(P + "text_ffn.0.bias")
            state_dict[P + "ffn.w_down_text"] = state_dict.pop(P + "text_ffn.2.weight")
            state_dict[P + "ffn.b_down_text"] = state_dict.pop(P + "text_ffn.2.bias")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._convert_grouped_state_dict(state_dict, prefix)
        direct_key = prefix + "text_modulation"
        linear_bias_key = prefix + "text_modulation.linear.bias"
        linear_weight_key = prefix + "text_modulation.linear.weight"
        old_empty_key = prefix + "text_empty_embedding"
        if self.text_modulation_timestep_dependent:
            if linear_bias_key not in state_dict and direct_key in state_dict:
                converted = state_dict.pop(direct_key)
                state_dict[linear_bias_key] = converted
                state_dict[linear_weight_key] = torch.zeros(
                    self.text_modulation.linear.weight.shape,
                    device=converted.device,
                    dtype=converted.dtype,
                )
            state_dict.pop(old_empty_key, None)
        else:
            if direct_key not in state_dict and linear_bias_key in state_dict:
                converted = state_dict[linear_bias_key]
                if linear_weight_key in state_dict and old_empty_key in state_dict:
                    old_weight = state_dict[linear_weight_key]
                    old_empty = torch.nn.functional.silu(
                        state_dict[old_empty_key].to(device=old_weight.device, dtype=old_weight.dtype)
                    )
                    converted = torch.matmul(old_weight, old_empty) + converted.to(
                        device=old_weight.device,
                        dtype=old_weight.dtype,
                    )
                state_dict[direct_key] = converted
            for old_key in (linear_bias_key, linear_weight_key, old_empty_key):
                state_dict.pop(old_key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward_packed_concat(
        self,
        x: torch.Tensor,
        ln_stats: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_concat: torch.Tensor,
        cu_seqlens_matmul: torch.Tensor,
        indices_interleave_to_concat: torch.Tensor,
        indices_concat_to_interleave: torch.Tensor,
        joint_cu_seqlens: torch.Tensor,
        max_seqlen: int,
        joint_packed_length: int,
        cp_joint_total: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if freqs_cis is None:
            raise ValueError("Packed JointAttentionBlock requires precomputed packed freqs_cis.")
        assert not self.text_modulation_timestep_dependent, (
            "Packed JointAttentionBlock assumes timestep-independent text modulation."
        )
        # pre_norm / norm2 are fused into packed_modulate via ln_stats; guard against a
        # silent change of norm type (the fused kernel hardcodes affine-free
        # LayerNorm with eps=1e-6).
        for _norm in (self.pre_norm, self.norm2):
            assert isinstance(_norm, nn.LayerNorm) and not _norm.elementwise_affine and _norm.eps == 1e-6, (
                "Packed JointAttentionBlock fuses pre_norm/norm2 as affine-free LayerNorm(eps=1e-6)."
            )

        # Channel-concat carries clean i2v/v2v context in input channels, so vec is
        # one noisy per-sample timestep embedding per video and modulates the video
        # tokens directly. Text modulation is timestep-independent (learned
        # parameter, inlined here), so all caption tokens share ONE text segment /
        # one modulation row.
        video_vec = vec
        # Merge per-modality modulation into one [video segments | 1 text segment]
        # tensor, matching cu_seqlens_concat (videos first, then the merged text),
        # then split once into the 6 modulation params (chunk returns views). q/k
        # norm has modality-specific affine params and is kept separate below.
        video_mod = self.modulation(video_vec)                                # [k, 6 * hidden]
        text_mod = self.text_modulation.unsqueeze(0).to(dtype=video_mod.dtype)  # [1, 6 * hidden]
        (
            mod1_shift, mod1_scale, mod1_gate,
            mod2_shift, mod2_scale, mod2_gate,
        ) = torch.cat([video_mod, text_mod], dim=0).chunk(6, dim=-1)

        # pre_norm fused into packed_modulate via the incoming ln_stats (LayerNorm
        # stats of x from the previous block / packed_simple_concat).
        x_mod = packed_modulate(
            x, shift=mod1_shift, scale=mod1_scale, cu_seqlens=cu_seqlens_concat, ln_stats=ln_stats,
        )

        # qkv_proj fuses linear -> per-head RMSNorm (Q/K) -> permute -> RoPE
        # (Q/K) into one op; q/k/v come back directly in interleave layout,
        # ready for flash attention.
        q, k, v = self.qkv_proj(
            x_mod, cu_seqlens_matmul, indices_interleave_to_concat, indices_concat_to_interleave, freqs_cis,
        )

        if cp_joint_total is not None:
            # Ulysses cp: q/k/v are this rank's strided token shard with all
            # heads. The wrapper all-to-alls tokens<->heads (and permutes
            # back to the real sequence order), then runs the varlen op on
            # the FULL sequence for H/cp heads against the full-sequence
            # joint_cu_seqlens -- numerically the single-GPU attention.
            attn = ulysses_flash_attn_varlen(
                q, k, v,
                cu_seqlens=joint_cu_seqlens,
                max_seqlen=max_seqlen,
                joint_total=cp_joint_total,
                group=current_parallel_layout().cp_group,
            )  # [local_joint_length, H*D]
        else:
            attn = flash_attn_varlen(
                q, k, v,
                cu_seqlens_q=joint_cu_seqlens,
                cu_seqlens_k=joint_cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
            )  # [joint_packed_length, H*D]

        # out_proj fuses the interleave->concat gather with the per-modality
        # projection (a single quack varlen_m gemm): the gather index is the
        # *whole* concat<->interleave permutation (not sliced to total_active).
        # cu_seqlens_matmul is passed as-is (last boundary = total_active, not
        # extended to joint_packed_length): the padding tail of the output buffer
        # is left uninitialized rather than covered by a segment, which is fine
        # since nothing downstream reads those rows. Default to this (no
        # extension) rather than padding cu_seqlens_m out to joint_packed_length.
        attn_out = self.out_proj(attn, indices_interleave_to_concat, cu_seqlens_matmul)
        # The gate fuses the residual add (bias=x) and emits ln_stats for norm2.
        x, ln_stats = packed_gate(
            attn_out, gate=mod1_gate, cu_seqlens=cu_seqlens_concat, bias=x, output_ln_stats=True,
        )

        # norm2 fused into packed_modulate via ln_stats.
        x_ffn_in = packed_modulate(
            x, shift=mod2_shift, scale=mod2_scale, cu_seqlens=cu_seqlens_concat, ln_stats=ln_stats,
        )
        ffn_out = self.ffn(x_ffn_in, cu_seqlens_matmul)
        # Fuse the residual add and emit ln_stats for the next block's pre_norm.
        x, ln_stats = packed_gate(
            ffn_out, gate=mod2_gate, cu_seqlens=cu_seqlens_concat, bias=x, output_ln_stats=True,
        )

        return x, ln_stats

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        vec: Optional[torch.Tensor] = None,
        ln_stats: Optional[torch.Tensor] = None,
        txt_len: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        max_seqlen: Optional[int] = None,
        joint_cu_seqlens: Optional[torch.Tensor] = None,
        joint_packed_length: Optional[int] = None,
        cu_seqlens_concat: Optional[torch.Tensor] = None,
        cu_seqlens_matmul: Optional[torch.Tensor] = None,
        indices_interleave_to_concat: Optional[torch.Tensor] = None,
        indices_concat_to_interleave: Optional[torch.Tensor] = None,
        cp_joint_total: Optional[int] = None,
    ) -> torch.Tensor:
        if cu_seqlens_matmul is not None:
            # Packed/training path: x is the concat layout [video | text | padding].
            if x is None or vec is None:
                raise ValueError("Packed JointAttentionBlock requires x and vec.")
            if ln_stats is None:
                raise ValueError("Packed JointAttentionBlock requires input ln_stats (LayerNorm stats).")
            if cu_seqlens_concat is None or joint_cu_seqlens is None:
                raise ValueError("Packed JointAttentionBlock requires cu_seqlens_concat and joint_cu_seqlens.")
            if indices_interleave_to_concat is None or indices_concat_to_interleave is None:
                raise ValueError("Packed JointAttentionBlock requires interleave<->concat indices.")
            if max_seqlen is None or joint_packed_length is None:
                raise ValueError("Packed JointAttentionBlock requires max_seqlen and joint_packed_length.")
            # Returns (x, ln_stats): the output ln_stats feeds the next block's pre_norm.
            return self.forward_packed_concat(
                x=x,
                ln_stats=ln_stats,
                vec=vec,
                freqs_cis=freqs_cis,
                cu_seqlens_concat=cu_seqlens_concat,
                cu_seqlens_matmul=cu_seqlens_matmul,
                indices_interleave_to_concat=indices_interleave_to_concat,
                indices_concat_to_interleave=indices_concat_to_interleave,
                joint_cu_seqlens=joint_cu_seqlens,
                max_seqlen=max_seqlen,
                joint_packed_length=joint_packed_length,
                cp_joint_total=cp_joint_total,
            )
        if x is None or vec is None:
            raise ValueError("JointAttentionBlock.forward requires x and vec for non-packed mode.")
        if txt_len is None:
            raise ValueError("Non-packed JointAttentionBlock requires txt_len.")

        (
            video_mod1_shift,
            video_mod1_scale,
            video_mod1_gate,
            video_mod2_shift,
            video_mod2_scale,
            video_mod2_gate,
        ) = self.modulation(vec).chunk(6, dim=-1)
        (
            text_mod1_shift,
            text_mod1_scale,
            text_mod1_gate,
            text_mod2_shift,
            text_mod2_scale,
            text_mod2_gate,
        ) = self._text_modulation(vec.shape[0], vec, vec).chunk(6, dim=-1)

        img = x[:, :-txt_len, :]
        txt = x[:, -txt_len:, :]
        # pre_norm / norm2 are affine-free and shared across video and text.
        img_mod = modulate(self.pre_norm(img), shift=video_mod1_shift, scale=video_mod1_scale)
        txt_mod = modulate(self.pre_norm(txt), shift=text_mod1_shift, scale=text_mod1_scale)

        img_q, img_k, img_v = self.qkv_proj.project(img_mod, "video")
        txt_q, txt_k, txt_v = self.qkv_proj.project(txt_mod, "text")

        n_vid = img_q.shape[1]
        video_freqs = (freqs_cis[0][:n_vid], freqs_cis[1][:n_vid])
        img_q, img_k = apply_rope(img_q, img_k, video_freqs)
        if freqs_cis[0].shape[0] > n_vid:
            # 4-axis joint RoPE carries extra text rows: rotate caption q/k too,
            # matching packed training (forward_channel_concat rotates the joint
            # video+text sequence). 3-axis models have no extra rows -> text
            # stays un-rotated, preserving prior behavior.
            text_freqs = (freqs_cis[0][n_vid:], freqs_cis[1][n_vid:])
            txt_q, txt_k = apply_rope(txt_q, txt_k, text_freqs)
        attn = joint_attention(
            video_qkv=(img_q, img_k, img_v),
            text_qkv=(txt_q, txt_k, txt_v),
            text_mask=text_mask,
            backend=self.attention_backend,
        )
        img_attn, txt_attn = attn.split([n_vid, attn.shape[1] - n_vid], dim=1)

        img = img + gate(self.out_proj.project(img_attn, "video"), video_mod1_gate)
        txt = txt + gate(self.out_proj.project(txt_attn, "text"), text_mod1_gate)

        img_ffn_in = modulate(self.norm2(img), shift=video_mod2_shift, scale=video_mod2_scale)
        txt_ffn_in = modulate(self.norm2(txt), shift=text_mod2_shift, scale=text_mod2_scale)
        img = img + gate(self.ffn.project(img_ffn_in, "video"), video_mod2_gate)
        txt = txt + gate(self.ffn.project(txt_ffn_in, "text"), text_mod2_gate)

        return torch.cat([img, txt], dim=1)


@dataclass
class PackedSequenceParams:
    # Caption-token boundaries after cropping the Qwen output down to user text only.
    caption_cu_seqlens: torch.Tensor
    video_cu_seqlens: torch.Tensor
    caption_segment_limit: int
    # One video+caption segment per sample, used by channel-concat attention.
    # Under context parallelism this one stays FULL-sequence (flash attention is
    # its only consumer, and Ulysses gathers the whole sequence before it), as
    # does caption_cu_seqlens (the caption encoder attends over whole captions);
    # every other boundary tensor here is in LOCAL, per-cp-rank coordinates.
    joint_cu_seqlens: torch.Tensor
    # Permutations between the two packed token layouts (each of length joint_packed_length,
    # padding included as identity so the mapping stays a pure permutation):
    #   interleave layout: v1, t1, v2, t2, ..., vn, tn, padding
    #   concat layout:     v1, v2, ..., vn, t1, t2, ..., tn, padding
    # concat = interleave[indices_interleave_to_concat]; interleave = concat[indices_concat_to_interleave].
    indices_interleave_to_concat: torch.Tensor
    indices_concat_to_interleave: torch.Tensor
    # Per-video segment boundaries plus one merged text segment in the concat layout,
    # shape [k + 2] (k video segments + 1 text segment). Text modulation is timestep-
    # independent, so all caption tokens share a single segment (packed_modulate /
    # packed_gate are per-token elementwise with a per-segment broadcast param).
    # The video segments align with cat([video_mod, single_text_mod]) in JointAttentionBlock.
    cu_seqlens_concat: torch.Tensor
    # Grouped-matmul boundaries over the concat layout: [0, total_video, total_video + total_caption], shape [3].
    cu_seqlens_matmul: torch.Tensor
    joint_segment_limit: int
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor
    # CUDA scalar consumed by Triton kernels; only the first active_sequence_count sequences are real.
    active_sequence_count: torch.Tensor
    # Total packed joint length seen by the single-stream backbone.
    joint_packed_length: int
    # Context parallelism (dataset.cp_batch.strided_split_batch): both are set
    # iff this batch is a cp shard. cp_text_index selects this rank's caption
    # rows -- text_states stays FULL so the caption encoder sees whole captions, and the
    # model gathers its output down afterwards. cp_joint_total is the FULL joint
    # length, which is what fixes the Ulysses shard sizes.
    cp_text_index: Optional[torch.Tensor] = None
    cp_joint_total: Optional[int] = None

    def mark_dynamic(self):
        for k in fields(self):
            value = getattr(self, k.name)
            if isinstance(value, torch.Tensor):
                torch._dynamo.mark_dynamic(value, 0)


# Architecture keys as configs from before the 2026 rename spell them -> constructor argument.
_LEGACY_ARCH_KEYS = {
    "heads_num": "num_attention_heads",
    "mm_single_blocks_depth": "num_layers",
    "mlp_act_type": "ffn_activation",
    "text_states_dim": "text_embed_dim",
    "rope_dim_list": "rope_axes_dim",
    "rope_t_interp": "rope_temporal_scale",
    "use_cond_type_embedding": "use_text_type_embedding",
    "qk_norm": "normalize_qk",
    "qk_norm_type": "qk_norm_kind",
    "attn_mode": "attention_backend",
    "use_attention_mask": "mask_text_padding",
}
# Legacy keys that only ever had one valid value; accepted with that value and dropped.
_LEGACY_FIXED_KEYS = {"mm_double_blocks_depth": 0, "single_stream_block_type": "mm_joint_attn"}
_LEGACY_WIDTH_KEYS = ("mlp_width_ratio", "video_mlp_width_ratio", "text_mlp_width_ratio")


def translate_architecture_config(config) -> dict:
    """Map an architecture config that may use the legacy key names onto the constructor arguments.

    Configs with the current names pass through unchanged. Legacy FFN widths follow their old
    rule: the video and text ratios default to ``mlp_width_ratio``, which defaulted to 4.
    """
    out, legacy, widths = {}, [], {}
    for key, value in dict(config).items():
        if key in _LEGACY_FIXED_KEYS:
            if value != _LEGACY_FIXED_KEYS[key]:
                raise ValueError(f"{key}={value!r} is not supported (only {_LEGACY_FIXED_KEYS[key]!r})")
            legacy.append(key)
        elif key in _LEGACY_WIDTH_KEYS:
            widths[key] = value
            legacy.append(key)
        elif key in _LEGACY_ARCH_KEYS:
            if _LEGACY_ARCH_KEYS[key] in config:
                raise ValueError(f"config sets both {key!r} and its new name {_LEGACY_ARCH_KEYS[key]!r}")
            out[_LEGACY_ARCH_KEYS[key]] = value
            legacy.append(key)
        else:
            out[key] = value
    if widths:
        base = widths.get("mlp_width_ratio", 4.0)
        for old, new in (("video_mlp_width_ratio", "mlp_ratio"), ("text_mlp_width_ratio", "context_mlp_ratio")):
            if new in config:
                raise ValueError(f"config sets both {old!r}/mlp_width_ratio and {new!r}")
            out[new] = base if widths.get(old) is None else widths[old]
    if legacy:
        loguru.logger.warning(f"Architecture config uses legacy key names {sorted(legacy)}; translated to the current ones.")
    return out


class TextTypeEmbedding(nn.Module):
    """Learned vector added to every caption token.

    ``weight`` has three rows of which only row 0 is read: ``(3, width)`` is the tensor
    layout the trained checkpoints store.
    """

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, width))

    @property
    def vector(self) -> torch.Tensor:
        return self.weight[0]


class PanVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """PAN 2 video diffusion transformer.

    Video latents are cut into ``patch_size`` (t, h, w) patches and the Qwen caption
    states are projected (RMSNorm + linear) and refined by a small self-attention
    stack; ``num_layers`` joint blocks then attend over the concatenated video+text
    sequence, and a modulated head maps the video tokens back to latent patches. The
    timestep conditions every block through adaptive-norm modulation.

    Constructor arguments are the model config (``model.transformer_config_overrides``
    in the training YAML); the defaults describe the released 29B model. Notable ones:

    * ``in_channels`` / ``out_channels``: latent channels in and out; ``in_channels``
      is ``2 * out_channels + 1`` when the clean condition and its mask are stacked
      onto the noisy latent.
    * ``mlp_ratio`` / ``context_mlp_ratio``: FFN width of the video / text tokens.
    * ``rope_axes_dim``: head-dim split across the (t, h, w) rotary axes; must sum to
      ``hidden_size // num_attention_heads``. ``rope_theta`` is the RoPE base and
      ``rope_temporal_scale`` multiplies frame positions (reference fps / data fps).
    * ``use_text_type_embedding``: add a learned vector to every text token.
    * ``zero_init_modulation``: start every modulation projection at zero.

    Configs using the legacy key names are translated by
    :func:`translate_architecture_config` (also applied by ``from_config``).
    """

    @register_to_config
    def __init__(
        self,
        # tokens / latents
        patch_size=(1, 2, 2), in_channels=97, out_channels=48,
        # backbone
        hidden_size=4096, num_attention_heads=32, num_layers=44,
        mlp_ratio=8.0, context_mlp_ratio=4.0, ffn_activation="gelu_tanh",
        # attention / positions
        attention_backend="flash", mask_text_padding=True, normalize_qk=True, qk_norm_kind="rms",
        rope_axes_dim=(16, 56, 56), rope_theta=256, rope_temporal_scale=16 / 24,
        # conditioning
        text_embed_dim=4096, use_text_type_embedding=True,
        zero_init_modulation=True, text_modulation_timestep_dependent=False,
    ):
        super().__init__()
        head_dim, leftover = divmod(hidden_size, num_attention_heads)
        if leftover:
            raise ValueError(f"hidden_size {hidden_size} is not a multiple of num_attention_heads {num_attention_heads}")
        if len(rope_axes_dim) != 3 or sum(rope_axes_dim) != head_dim:
            raise ValueError(f"rope_axes_dim {rope_axes_dim} needs three (t, h, w) entries summing to the head dim {head_dim}")

        self.patch_size = list(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.hidden_size, self.num_attention_heads = hidden_size, num_attention_heads
        # Frame positions are multiplied by rope_temporal_scale = reference fps / data fps,
        # so one RoPE time unit stays a fixed wall-clock duration when the data fps changes.
        self.rope_axes_dim, self.rope_theta, self.rope_temporal_scale = list(rope_axes_dim), rope_theta, rope_temporal_scale
        self.attention_backend, self.mask_text_padding = attention_backend, mask_text_padding
        self.text_embed_dim = text_embed_dim
        self.text_modulation_timestep_dependent = text_modulation_timestep_dependent

        # Construction order below is the parameter (and initializer RNG) order; keep it.
        self.patch_embed = VideoPatchEmbed(self.patch_size, self.in_channels, hidden_size)
        # Caption path: RMSNorm + linear into the backbone width, then the caption encoder.
        self.cap_embedder = nn.Sequential(RMSNorm(text_embed_dim, eps=1e-5), nn.Linear(text_embed_dim, hidden_size))
        self.caption_encoder = CaptionEncoder(hidden_size, num_attention_heads, num_layers=4)
        self.timestep_encoder = TimestepEncoder(hidden_size, activation_factory("silu"))
        block_options = dict(
            attention_backend=attention_backend, normalize_qk=normalize_qk, qk_norm_kind=qk_norm_kind,
            ffn_activation=ffn_activation, mlp_ratio=mlp_ratio, context_mlp_ratio=context_mlp_ratio,
            zero_init_modulation=zero_init_modulation,
            text_modulation_timestep_dependent=text_modulation_timestep_dependent,
        )
        self.blocks = nn.ModuleList(
            [JointAttentionBlock(hidden_size, num_attention_heads, **block_options) for _ in range(num_layers)]
        )
        self.output_head = OutputHead(hidden_size, self.patch_size, self.out_channels, activation_factory("silu"),
                                      zero_init_modulation=zero_init_modulation)
        self.text_type_embedding = TextTypeEmbedding(hidden_size) if use_text_type_embedding else None

    @classmethod
    def from_config(cls, config=None, **kwargs):
        if config is not None:
            meta = {k: v for k, v in dict(config).items() if k.startswith("_")}
            arch = translate_architecture_config({k: v for k, v in dict(config).items() if not k.startswith("_")})
            config = {**meta, **arch}
        return super().from_config(config, **kwargs)

    def rope_settings(self):
        """``(axes_dim, theta, per-axis position scale)`` of the rotary embedding; only frames are rescaled."""
        axes_dim = tuple(self.rope_axes_dim)
        return axes_dim, self.rope_theta, (float(self.rope_temporal_scale),) + (1.0,) * (len(axes_dim) - 1)

    def rope_tables(self, grid):
        """(cos, sin) RoPE tables for a dense ``(t, h, w)`` token grid, in token order."""
        axes_dim, theta, position_scale = self.rope_settings()
        return rope_tables_for_grid(axes_dim, tuple(grid), theta=theta, position_scale=position_scale)

    def validate_stg_layers(self, stg_skip_layers):
        """Validate whole-block STG for dense, single-device T2V inference only."""
        if stg_skip_layers is None:
            return ()
        if not isinstance(stg_skip_layers, (list, tuple)) or not stg_skip_layers:
            raise ValueError("stg_skip_layers must be a nonempty list/tuple of zero-based block indices.")
        if any(type(index) is not int for index in stg_skip_layers):
            raise ValueError("stg_skip_layers must contain integer block indices (not bool).")
        if len(set(stg_skip_layers)) != len(stg_skip_layers):
            raise ValueError("stg_skip_layers must not contain duplicate block indices.")
        if any(index < 0 or index >= len(self.blocks) for index in stg_skip_layers):
            raise ValueError(f"stg_skip_layers must be in [0, {len(self.blocks) - 1}].")
        if self.training or torch.is_grad_enabled():
            raise ValueError("STG requires model.eval() and torch.no_grad()/inference_mode().")
        if self.attention_backend not in {"flash", "flash4", "torch"}:
            raise ValueError("STG supports dense attention only.")
        if current_parallel_layout().cp_enabled:
            raise ValueError("STG does not support context parallelism.")
        if any(
            getattr(module, name, None) is not None
            for module in self.modules()
            for name in ("_hf_hook", "_diffusers_hook", "cache_helper")
        ):
            raise ValueError("STG does not support offloading or cache hooks.")
        return tuple(stg_skip_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        return_dict: bool = False,
        packed_seq_params: PackedSequenceParams = None,
        target: Optional[torch.Tensor] = None,
        cp_meta: Optional[Any] = None,
        stg_skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if stg_skip_layers is not None:
            # STG weak pass: skip whole joint blocks. Inference-only, so it is
            # rejected for the packed training path and every extra conditioning.
            stg_skip_layers = self.validate_stg_layers(stg_skip_layers)
            if packed_seq_params is not None or target is not None:
                raise ValueError("STG supports only the unpacked T2V forward, not packed training.")
        packed_mode_enabled = packed_seq_params is not None
        parallel_dims = current_parallel_layout()
        cp_enabled = parallel_dims.cp_enabled
        # cp_meta (dataset.packed_dataloader.CpMeta) is dataloader-side state for
        # dataset.cp_batch; it rides on PackedBatch (so it arrives here via
        # **batch.__dict__) and is None on every cp shard. The model never reads
        # it; still set under cp means a FULL batch reached the model unsharded.
        if cp_enabled and cp_meta is not None:
            raise ValueError(
                "PackedBatch.cp_meta is set: under cp the batch must go through "
                "dataset.cp_batch.split_packed_batch_for_cp before the model.")
        if packed_mode_enabled:
            # Packed-mode input variables:
            # - hidden_states: packed video latent tensor [C, T_total_img, H, W].
            # - text_states: packed text token tensor [T_total_txt, text_states_dim].
            # - timestep: [N] (N is number of sequences, or max number of sequences if padded)
            # - text_mask: must be None (captions are delimited by cu_seqlens)
            # - packed_seq_params: explicit packed metadata with segment boundaries and indices.
            # - target: same shape as hidden_states
            # Output:
            # - mse_loss instead of pred
            if cp_enabled != (packed_seq_params.cp_joint_total is not None):
                raise ValueError(
                    "cp_size and the batch disagree: cp batches must come from "
                    "dataset.cp_batch.strided_split_batch, and non-cp batches "
                    "must not."
                )
            if text_mask is not None:
                raise ValueError("Packed mode requires text_mask=None.")

            vec = self.timestep_encoder(timestep).to(hidden_states)
            txt = self.cap_embedder(text_states)
            txt = self.caption_encoder(
                txt,
                cu_seqlens=packed_seq_params.caption_cu_seqlens,
                max_seqlen=packed_seq_params.caption_segment_limit,
            )
            if packed_seq_params.cp_text_index is not None:
                # cp: the caption encoder above ran on the FULL captions (its attention
                # needs them whole, and captions are small enough to refine
                # redundantly on every cp rank); everything downstream is
                # per-token, so gather this rank's caption rows here.
                txt = txt.index_select(0, packed_seq_params.cp_text_index)
            vid = self.patch_embed(hidden_states, packed=True)

            text_type_vec = None if self.text_type_embedding is None else self.text_type_embedding.vector  # [C]

            # Concat layout: [v0, v1, ..., vn, t0, ..., tn, padding]. text_type_vec is
            # broadcast-added to the text block inside the kernel. ln_stats carries the
            # LayerNorm stats fused into the first block's pre_norm.
            x, ln_stats = packed_simple_concat(
                vid, txt,
                cu_seqlens_matmul=packed_seq_params.cu_seqlens_matmul,
                total_seqlen=packed_seq_params.joint_packed_length,
                text_type_vec=text_type_vec,
                output_ln_stats=True,
            )

            for index, block in enumerate(self.blocks):
                # Each block consumes ln_stats (pre_norm stats) and returns the
                # ln_stats for the next block's pre_norm (fused gate -> norm).
                x, ln_stats = block(
                    x=x,
                    vec=vec,
                    ln_stats=ln_stats,
                    freqs_cis=(packed_seq_params.freqs_cos, packed_seq_params.freqs_sin),
                    cu_seqlens_concat=packed_seq_params.cu_seqlens_concat,
                    cu_seqlens_matmul=packed_seq_params.cu_seqlens_matmul,
                    indices_interleave_to_concat=packed_seq_params.indices_interleave_to_concat,
                    indices_concat_to_interleave=packed_seq_params.indices_concat_to_interleave,
                    joint_cu_seqlens=packed_seq_params.joint_cu_seqlens,
                    max_seqlen=packed_seq_params.joint_segment_limit,
                    joint_packed_length=packed_seq_params.joint_packed_length,
                    cp_joint_total=packed_seq_params.cp_joint_total,
                )

            # OutputHead is token-wise, so modulating only the video block (the
            # leading rows of the concat layout) is numerically identical to
            # modulating the joint sequence. Video tokens use the per-sample noisy
            # timestep modulation vec, with one segment per video (video_cu_seqlens).
            # ln_stats fuses the head norm for the video tokens (same stats).
            x = self.output_head(
                x, vec, cu_seqlens=packed_seq_params.video_cu_seqlens, ln_stats=ln_stats,
                cu_seqlens_matmul=packed_seq_params.cu_seqlens_matmul,
            )

            loss = packed_concat_mse_loss(
                x, target,
                cu_seqlens_matmul=packed_seq_params.cu_seqlens_matmul,
            )

            return loss

        return self._forward_dense(hidden_states, timestep, text_states, text_mask,
                                   freqs_cos, freqs_sin, stg_skip_layers, return_dict)

    def _encode_text(self, text_states, text_mask):
        """Caption tokens in the backbone width: projection, caption encoder, text-type vector."""
        mask = text_mask if self.mask_text_padding else None
        text = self.caption_encoder(self.cap_embedder(text_states), mask=mask)
        if self.text_type_embedding is not None:
            text = text + self.text_type_embedding.vector
        return text

    def _forward_dense(self, latents, timestep, text_states, text_mask, freqs_cos, freqs_sin,
                       skip_blocks=None, return_dict=False):
        """Unpacked forward used at inference.

        ``latents`` is ``[B, C, T, H, W]``; returns ``(velocity, None)`` with the
        velocity in the same layout. ``skip_blocks`` (STG) drops whole joint blocks.
        """
        if return_dict:
            raise ValueError("return_dict=True is not supported")
        grid = tuple(size // patch for size, patch in zip(latents.shape[2:], self.patch_size))
        rope = self.rope_tables(grid) if freqs_cos is None and freqs_sin is None else (freqs_cos, freqs_sin)

        video = self.patch_embed(latents)
        temb = self.timestep_encoder(timestep)
        text = self._encode_text(text_states, text_mask)
        hidden, n_video = torch.cat((video, text), dim=1), video.shape[1]
        for index, block in enumerate(self.blocks):
            if not (skip_blocks and index in skip_blocks):
                hidden = block(x=hidden, vec=temb, txt_len=text.shape[1], freqs_cis=rope, text_mask=text_mask)
        velocity = self.unpatchify(self.output_head(hidden[:, :n_video], temb), *grid)
        return velocity, None

    def unpatchify(self, tokens, t, h, w):
        """Fold ``[N, t*h*w, C*pt*ph*pw]`` patch tokens back into a ``[N, C, t*pt, h*ph, w*pw]`` latent.

        Token order is (t, h, w) row-major and each token's features are laid out
        (C, pt, ph, pw), matching ``VideoPatchEmbed`` / the packed dataloader.
        """
        if tokens.shape[1] != t * h * w:
            raise ValueError(f"expected {t * h * w} tokens for grid {(t, h, w)}, got {tokens.shape[1]}")
        pt, ph, pw = self.patch_size
        return rearrange(
            tokens, "n (t h w) (c pt ph pw) -> n c (t pt) (h ph) (w pw)",
            t=t, h=h, w=w, c=self.out_channels, pt=pt, ph=ph, pw=pw,
        )
