import torch
from torch import nn
import triton
import triton.language as tl

import aiter
from atom.utils.context import get_context
from flash_attn import flash_attn_with_kvcache


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                min_seqlen_q = context.min_seqlen_q,
                dropout_p=context.dropout_p,
                softmax_scale=self.scale,
                causal=True,
            )
        else:  # decode
            # cu_seqlens_q=torch.tensor([0, q.shape[0]],
            #                           dtype=torch.int32,
            #                           device=q.device),
            # cu_seqlens_k=torch.tensor([0, k],
            #                           dtype=torch.int32,
            #                           device=k.device),
            # max_seqlen_q=q.shape[1]
            # max_seqlen_k = k.shape[1]
            o = aiter.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=0,
                min_seqlen_q = context.min_seqlen_q,
                dropout_p=context.dropout_p,
                softmax_scale=self.scale,
                causal=True,
            )

            # o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            #                             cache_seqlens=context.context_lens, block_table=context.block_tables,
            #                             softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
