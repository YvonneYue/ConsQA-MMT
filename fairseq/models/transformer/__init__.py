# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .transformer_decoder import TransformerDecoder, TransformerDecoderBase, Linear
from .transformer_encoder import TransformerEncoder, TransformerEncoderBase
from .transformer_legacy import (
    TransformerModel,
    base_architecture,
    tiny_architecture,
    transformer_iwslt_de_en,
    transformer_wmt_en_de,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big,
    transformer_wmt_en_de_big,
    transformer_wmt_en_de_big_t2t,
)
from .transformer_base import TransformerModelBase, Embedding
from .transformer_mmt_vqa_legacy import TransformerMMTVQAModel, transformer_mmt_vqa_2sa_2decoder, mmt_vqa_base_architecture
from .transformer_mmt_vqa2_legacy import TransformerMMTVQA2Model, transformer_mmt_vqa2_2sa_2decoder, mmt_vqa2_base_architecture
from .transformer_mmt_legacy import TransformerMMTModel, transformer_mmt_dual_encoder, mmt_base_architecture, transformer_mmt_single_encoder
from .transformer_mmt_nmt_vqa2_legacy import TransformerMMTandNMTVQA2Model, transformer_mmt_nmt_vqa2_2sa_2decoder,mmt_nmt_vqa2_base_architecture
from .transformer_consqa_mmt_legacy import Transformer_ConsQA_MMT_Model, transformer_consqa_mmt_2sa_2decoder, consqa_mmt_base_architecture
__all__ = [
    "TransformerModelBase",
    "TransformerConfig",
    "TransformerDecoder",
    "TransformerDecoderBase",
    "TransformerEncoder",
    "TransformerEncoderBase",
    "TransformerModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    # MMT_VQA
    "TransformerMMTVQAModel",
    "TransformerMMTVQA2Model",
    "transformer_mmt_vqa_2sa_2decoder",
    "transformer_mmt_vqa2_2sa_2decoder",
    "mmt_vqa_base_architecture",
    "mmt_vqa2_base_architecture",
    "transformer_mmt_nmt_vqa2",
    "transformer_mmt_nmt_vqa2_2sa_2decoder",
    "consqa_mmt_base_architecture",
    "transformer_consqa_mmt"
    "transformer_consqa_mmt_2sa_2decoder",
    # MMT
    "TransformerMMTModel",
    "transformer_mmt_dual_encoder",
    "transformer_mmt_single_encoder",
    "mmt_base_architecture",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
