from dataclasses import dataclass
from typing import Type, Optional

from atom.config import KVCacheConfig, KVCacheTensor
from atom.utils.forward_context import AttentionMetaData, Context
from .backends import CommonAttentionBuilder, AttentionBackend
import torch
import numpy as np

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.config import get_current_atom_config
from aiter import get_mla_metadata_v1, get_mla_metadata_info_v1, dtypes
from aiter.dist.parallel_state import get_tp_group

from .backends import AttentionBackend, CommonAttentionBuilder


class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


class AiterMLAMetadataBuilder(CommonAttentionBuilder):

    def __init__(self, block_size: int, device: torch.device):
        super().__init__(
            block_size,
            device,
        )  # Call parent __init__ to initialize _cached_kv_cache_data
        assert self.block_size == 1, "AITER MLA requires only block size 1."
        config = get_current_atom_config()
        hf_config = config.hf_config
        self.num_attention_heads = (
            hf_config.num_attention_heads // get_tp_group().world_size
        )
        self.is_sparse = hasattr(hf_config, "index_topk")
        self.index_topk = hf_config.index_topk if self.is_sparse else -1

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            config.max_num_seqs,
            1,
            self.num_attention_heads,
            torch.bfloat16,
            dtypes.d_dtypes[config.kv_cache_dtype],
            is_sparse=self.is_sparse,
            fast_mode=True,
        )
        # AITER MLA specific persistent buffers
        self.work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self.work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self.work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )

        self.reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self.reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self.reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )

    def set_mla_persistent_worker_buffers(self, forward_vars, bs: int):
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": 1,
            "uni_seqlen_qo": 1,
            "fast_mode": 1,
            "topk": -1,
        }
        get_mla_metadata_v1(
            forward_vars["cu_seqlens_q"].gpu[: bs + 1],
            (
                forward_vars["sparse_kv_indptr"].gpu[: bs + 1]
                if self.is_sparse
                else forward_vars["kv_indptr"].gpu[: bs + 1]
            ),
            self.num_attention_heads,
            1,  # nhead_kv,
            True,
            self.work_meta_data,
            self.work_info_set,
            self.work_indptr,
            self.reduce_indptr,
            self.reduce_final_map,
            self.reduce_partial_map,
            **split_params,
        )
        return {
            "work_meta_data": self.work_meta_data,
            "work_info_set": self.work_info_set,
            "work_indptr": self.work_indptr,
            "reduce_indptr": self.reduce_indptr,
            "reduce_final_map": self.reduce_final_map,
            "reduce_partial_map": self.reduce_partial_map,
        }

    def prepare_decode(self, batch: ScheduledBatch, bs: int, forward_vars):
        scheduled_bs = batch.total_seqs_num_decode
        seqs = list(batch.seqs.values())
        dropout_p = 0.0
        max_q_len = 1

        context_lens = [seq.num_tokens for seq in seqs]
        positions = context_lens
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in seqs
        ]

        var = forward_vars
        # prepare_block_tables
        block_tables = var["block_tables"].np
        for i, seq in enumerate(seqs):
            block_tables[i] = 0
            block_tables[i, : seq.num_blocks] = seq.block_table

        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[:scheduled_bs] = slot_mapping
        var["slot_mapping"].np[scheduled_bs:bs] = -1
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens

        sum_blocks = 0
        for seq in seqs:
            var["kv_indices"].np[
                sum_blocks : sum_blocks + seq.num_blocks
            ] = seq.block_table
            sum_blocks += seq.num_blocks
        kv_indptr = np.cumsum([seq.num_blocks for seq in seqs])
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks
        var["kv_last_page_lens"].np[:scheduled_bs] = [
            seq.last_block_num_tokens for seq in seqs
        ]
        var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
            ("cu_seqlens_q", bs + 1),
            ("kv_indptr", bs + 1),
            ("kv_indices", sum_blocks),
            ("kv_last_page_lens", bs),
        ]
        if self.is_sparse:
            index_topk = self.index_topk
            sparse_context_lens = np.clip(var["context_lens"].np[:bs], None, index_topk)
            var["sparse_kv_indptr"].np[1 : bs + 1] = np.cumsum(
                sparse_context_lens, dtype=np.int32
            )
            var["sparse_kv_indptr"].np[scheduled_bs : bs + 1] = var[
                "sparse_kv_indptr"
            ].np[scheduled_bs]
            vars_used.append(("sparse_kv_indptr", bs + 1))

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        ctx_mla_ps = self.set_mla_persistent_worker_buffers(forward_vars, bs)
        ctx.update(ctx_mla_ps)
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        # if str(positions.device) == "cuda:0":
        #     for el, var in ctx.items():
        #         print(f"{el}: {var}")
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, forward_vars, bs: int) -> AttentionMetaData:
        sparse_kv_indptr = (
            forward_vars["sparse_kv_indptr"].gpu if self.is_sparse else None
        )
        ctx_mla_ps = self.set_mla_persistent_worker_buffers(forward_vars, bs)
        attn_matadata = AttentionMetaData(
            slot_mapping=forward_vars["slot_mapping"].gpu[:bs],
            context_lens=forward_vars["context_lens"].gpu[:bs],
            block_tables=forward_vars["block_tables"].gpu[:bs],
            max_q_len=1,
            cu_seqlens_q=forward_vars["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=forward_vars["kv_indptr"].gpu[: bs + 1],
            kv_indices=forward_vars["kv_indices"].gpu[:],
            kv_last_page_lens=forward_vars["kv_last_page_lens"].gpu[:bs],
            sparse_kv_indptr=sparse_kv_indptr,
            **ctx_mla_ps,
        )
        positions = forward_vars["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_matadata, context
