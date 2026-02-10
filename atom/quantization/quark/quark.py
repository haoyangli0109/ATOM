# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union, cast
from abc import abstractmethod


import torch
from atom.utils import envs, get_open_port
from atom.utils.distributed.utils import stateless_init_torch_distributed_process_group
from atom.config import QuantizationConfig

from torch.distributed import ProcessGroup, ReduceOp
from transformers import AutoConfig, PretrainedConfig, GenerationConfig

from aiter import QuantType
from aiter.dist.parallel_state import get_dp_group
from aiter.utility.dtypes import d_dtypes

logger = logging.getLogger("atom")

# TODO: Inherit from QuantizeMethodBase to implement online quantization, hybrid quantization
class QuarkConfig():

    def __init__(
        self,
        orig_quant_config: dict[str, Any]
    ):
        self.exclude_layers = None
        self.quant_config = self.parse_gloabal_quant_config(orig_quant_config=orig_quant_config)

    def parse_gloabal_quant_config(self, orig_quant_config: dict[str, Any]) -> QuantizationConfig:
        # preprocess
        self.exclude_layers = cast(list[str], orig_quant_config.get("exclude"))
        if orig_quant_config.get("layer_quant_config") is not None:
            logger.warning(
                "layer_quant_config is not supported yet and will be ignored"
            )

        if orig_quant_config.get("layer_type_quant_config") is not None:
            logger.warning(
                "layer_type_quant_config is not supported yet and will be ignored"
            )
        config = cast(dict[str, Any], orig_quant_config.get("global_quant_config"))
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        # parse quark config
        weight_config = cast(dict[str, Any], config.get("weight"))
        # The default quantization type for W and A in atom is currently consistent.
        input_config = cast(dict[str, Any], config.get("input_tensors"))
        qscheme = weight_config.get("qscheme")

        # quant_type
        if qscheme == "per_group":
            group_size = weight_config.get("group_size")
            if group_size == 128:
                quant_type = QuantType.per_1x128
            elif group_size == 32:
                quant_type = QuantType.per_1x32
            else:
                logger.error(f"Unsupported group size {group_size}")
        elif qscheme == "per_channel":
            quant_type = QuantType.per_Token
        else:
            quant_type = QuantType.per_Tensor

        # quant_dtype
        q_dtype = weight_config.get("dtype")
        # atom accepts fp8 instead of fp8_e4m3
        q_dtype = q_dtype.split("_")[0]
        if "fp4" in q_dtype:
            q_dtype = "fp4x2"
        quant_dtype = d_dtypes[q_dtype]

        # dynamic or not
        is_dynamic = input_config.get("is_dynamic")

        return QuantizationConfig(
            quant_type, quant_dtype, is_dynamic, quant_method="quark"
        )

