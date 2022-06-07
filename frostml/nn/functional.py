from typing import Optional, Tuple

import math
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout, fold, linear, softmax


def in_projection(
        q: Tensor, k: Tensor, v: Tensor,
        w_q: Tensor, w_k: Tensor, w_v: Tensor,
        b_q: Optional[Tensor] = None, b_k: Optional[Tensor] = None, b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def scale_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tensor:
    B, Nt, E = q.shape
    scale = math.sqrt(E)
    q = q / scale
    # (B, Nt, E) @ (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    attn = attn + attn_mask if attn_mask is not None else attn
    attn = softmax(attn, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.)  # prevent nan resulting from padded-out inputs
    attn = dropout(attn, p=dropout_p) if dropout_p > 0.0 else attn
    # (B, Nt, Ns) @ (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output


def multi_head_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        q_proj_weight: Tensor,
        k_proj_weight: Tensor,
        v_proj_weight: Tensor,
        q_proj_bias: Optional[Tensor],
        k_proj_bias: Optional[Tensor],
        v_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        dropout_p: float,
        training: bool = True,
        attn_mask: Optional[Tensor] = None,
) -> Tensor:
    # set up shape vars
    bsz, nt, embed_dim = query.shape
    bsz, ns, _ = key.shape
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        dim_heads = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        dim_heads = embed_dim // num_heads

    # in projection
    q, k, v = in_projection(query, key, value,
                            q_proj_weight, k_proj_weight, v_proj_weight,
                            q_proj_bias, k_proj_bias, v_proj_bias)

    # reshape q, k, v for multi head attention
    q = q.contiguous().view(bsz * num_heads, nt, dim_heads)
    k = k.contiguous().view(bsz * num_heads, ns, dim_heads)
    v = v.contiguous().view(bsz * num_heads, ns, dim_heads)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # calculate attention
    attn_output = scale_dot_product_attention(q, k, v, attn_mask, dropout_p)

    # out projection
    attn_output = attn_output.contiguous().view(bsz, nt, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output


def outlook_attention(
        x: Tensor,
        num_heads: int,
        v_proj_weight: Tensor,
        a_proj_weight: Tensor,
        v_proj_bias: Optional[Tensor],
        a_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        kernel_size: int,
        stride: int,
        unfold: Module,
        pool: Module,
        dropout_p: float,
        padding: int,
        training: bool = True,
):
    B, C, H, W = x.shape
    scale = math.sqrt(C / num_heads)
    dropout_p = 0.0 if not training else dropout_p
    v = linear(x.permute(0, 2, 3, 1), v_proj_weight, v_proj_bias).permute(0, 3, 1, 2)
    h, w = math.ceil(H / stride), math.ceil(W / stride)
    # B, H, N, KK, E
    v = unfold(v).contiguous().view(B, num_heads, C // num_heads, kernel_size ** 2, h * w).permute(0, 1, 4, 3, 2)
    attn = pool(x).permute(0, 3, 1, 2)
    attn = linear(attn, a_proj_weight, a_proj_bias)
    # B, H, N, KK, KK
    attn = attn.contiguous().view(B, h * w, num_heads, kernel_size ** 2, kernel_size ** 2).permute(0, 2, 1, 3, 4)
    attn = attn / scale
    attn = softmax(attn, dim=-1)
    attn = dropout(attn, p=dropout_p)
    output = torch.bmm(attn, v).permute(0, 1, 4, 3, 2)
    output = output.contiguous().view(B, C * kernel_size ** 2, h * w)
    output = fold(output, output_size=(H, W), kernel_size=kernel_size, padding=padding, stride=stride)
    output = linear(output.permute(0, 2, 3, 1), out_proj_weight, out_proj_bias)
    output = dropout(output, p=dropout_p)
    return output
