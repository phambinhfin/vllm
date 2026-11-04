# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``get_top_tokens`` on the ROCm DeepSeek-V3.2 / GLM-5.x MTP drafter.

Checks it selects the same token as the full-vocab argmax, including with a
padded vocabulary and across multiple MTP layers.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import PreTrainedConfig

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "get_top_tokens is implemented on the ROCm DeepSeek-V3.2 MTP drafter only.",
        allow_module_level=True,
    )

from vllm.models.deepseek_v32.amd.mtp import (  # noqa: E402
    DeepseekV32MTP,
    DeepseekV32MultiTokenPredictor,
)

DEVICE_TYPE = current_platform.device_type
START_LAYER_IDX = 61
HIDDEN_SIZE = 1024
RMS_NORM_EPS = 1e-6
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


def _make_predictor(vocab_size: int, num_mtp_layers: int = 1):
    config = PreTrainedConfig()
    config.hidden_size = HIDDEN_SIZE
    config.vocab_size = vocab_size
    config.rms_norm_eps = RMS_NORM_EPS

    generator = torch.Generator().manual_seed(7)
    layers = {}
    for i in range(num_mtp_layers):
        # single vocab shard: reduction is a local argmax, no process group
        with (
            mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=0),
            mock.patch(
                f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            shared_head = SharedHead(config=config, prefix=f"shared_head.{i}")
        with torch.no_grad():
            # distinct per-layer heads so wrong-layer routing changes the token
            shared_head.head.weight.normal_(0.0, 0.02, generator=generator)
        shared_head = shared_head.to(device=DEVICE_TYPE, dtype=torch.bfloat16)
        layers[str(START_LAYER_IDX + i)] = SimpleNamespace(shared_head=shared_head)

    predictor = object.__new__(DeepseekV32MultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.layers = layers
    predictor.mtp_start_layer_idx = START_LAYER_IDX
    predictor.num_mtp_layers = num_mtp_layers
    predictor.logits_processor = LogitsProcessor(vocab_size)
    return predictor


def _hidden_states(num_tokens: int, seed: int) -> torch.Tensor:
    """The post-``shared_head.norm`` hidden state the MTP layer returns."""
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.float32, generator=generator
    )
    return hidden_states.to(device=DEVICE_TYPE, dtype=torch.bfloat16)


def _assert_selects_max_logit(top: torch.Tensor, logits: torch.Tensor) -> None:
    """bf16 ties make comparing values safer than comparing indices."""
    assert top.shape == (logits.shape[0],)
    selected = logits.gather(1, top.unsqueeze(1)).squeeze(1)
    assert torch.equal(selected, logits.max(dim=-1).values)


def test_get_top_tokens_matches_full_argmax(default_vllm_config):
    """Same draft token as the path taken when the reduction is off."""
    predictor = _make_predictor(vocab_size=512)

    for num_tokens in (1, 16, 128):
        hidden_states = _hidden_states(num_tokens, seed=11)

        logits = predictor.compute_logits(hidden_states)
        top = predictor.get_top_tokens(hidden_states)

        assert top.dtype == torch.int64
        _assert_selects_max_logit(top, logits)


def test_get_top_tokens_honors_padded_vocab(default_vllm_config):
    """Padding entries above ``org_vocab_size`` must never win the argmax."""
    vocab_size = 500
    predictor = _make_predictor(vocab_size)
    head = predictor._shared_head(0).head
    assert head.shard_indices.num_org_vocab_padding > 0

    hidden_states = _hidden_states(32, seed=13)
    logits = predictor.compute_logits(hidden_states)
    top = predictor.get_top_tokens(hidden_states)

    _assert_selects_max_logit(top, logits)
    assert (top < vocab_size).all()


def test_get_top_tokens_routes_to_the_step_s_head(default_vllm_config):
    """``spec_step_idx`` must select the same head ``compute_logits`` uses."""
    predictor = _make_predictor(vocab_size=512, num_mtp_layers=2)
    hidden_states = _hidden_states(16, seed=17)

    for spec_step_idx in (0, 1, 2, 3):  # 2 and 3 exercise the modulo wrap
        logits = predictor.compute_logits(hidden_states, spec_step_idx)
        top = predictor.get_top_tokens(hidden_states, spec_step_idx)
        _assert_selects_max_logit(top, logits)

    # The two heads really are different, so the check above has teeth.
    top_0 = predictor.get_top_tokens(hidden_states, 0)
    top_1 = predictor.get_top_tokens(hidden_states, 1)
    assert not torch.equal(top_0, top_1)


def test_wrapper_forwards_spec_step_idx():
    """``DeepseekV32MTP`` is what the speculator probes, so it needs the method."""
    assert hasattr(DeepseekV32MTP, "get_top_tokens")

    hidden_states = torch.empty(0)
    wrapper = object.__new__(DeepseekV32MTP)
    wrapper.model = SimpleNamespace(get_top_tokens=lambda *args: args)

    assert DeepseekV32MTP.get_top_tokens(wrapper, hidden_states, 2) == (
        hidden_states,
        2,
    )
    assert DeepseekV32MTP.get_top_tokens(wrapper, hidden_states) == (hidden_states, 0)
