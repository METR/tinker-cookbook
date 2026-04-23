"""Tests for completer types and metadata propagation."""

import pytest

from tinker_cookbook.completers import SamplingMetadata, TokensWithLogprobs


class TestSamplingMetadata:
    def test_stop_reason_literal_values(self):
        """Test that SamplingMetadata accepts valid stop_reason values."""
        meta_stop = SamplingMetadata(stop_reason="stop")
        assert meta_stop.stop_reason == "stop"

        meta_length = SamplingMetadata(stop_reason="length")
        assert meta_length.stop_reason == "length"

    def test_optional_logprob_fields_default_to_none(self):
        """Test that optional logprob fields default to None."""
        meta = SamplingMetadata(stop_reason="stop")
        assert meta.prompt_logprobs is None
        assert meta.topk_prompt_logprobs is None

    def test_optional_logprob_fields_can_be_set(self):
        """Test that optional logprob fields can be set."""
        meta = SamplingMetadata(
            stop_reason="stop",
            prompt_logprobs=[0.1, 0.2, None, 0.3],
            topk_prompt_logprobs=[[(1, 0.5), (2, 0.3)], None, [(3, 0.9)]],
        )
        assert meta.prompt_logprobs == [0.1, 0.2, None, 0.3]
        assert meta.topk_prompt_logprobs is not None
        assert len(meta.topk_prompt_logprobs) == 3


class TestTokensWithLogprobs:
    def test_backward_compatible_construction(self):
        """Test that TokensWithLogprobs can be constructed without metadata."""
        twl = TokensWithLogprobs(tokens=[1, 2, 3], maybe_logprobs=[0.1, 0.2, 0.3])
        assert twl.tokens == [1, 2, 3]
        assert twl.maybe_logprobs == [0.1, 0.2, 0.3]
        assert twl.sampling_metadata is None
        assert twl.stop_reason is None

    def test_construction_with_metadata(self):
        """Test TokensWithLogprobs with sampling metadata."""
        meta = SamplingMetadata(stop_reason="length")
        twl = TokensWithLogprobs(
            tokens=[1, 2, 3],
            maybe_logprobs=[0.1, 0.2, 0.3],
            sampling_metadata=meta,
        )
        assert twl.sampling_metadata is meta
        assert twl.stop_reason == "length"

    def test_stop_reason_property_with_none_metadata(self):
        """Test stop_reason property when metadata is None."""
        twl = TokensWithLogprobs(tokens=[1], maybe_logprobs=None)
        assert twl.stop_reason is None

    def test_logprobs_property_raises_when_none(self):
        """Test logprobs property raises ValueError when maybe_logprobs is None."""
        twl = TokensWithLogprobs(tokens=[1], maybe_logprobs=None)
        with pytest.raises(ValueError, match="Logprobs are not available"):
            _ = twl.logprobs

    def test_logprobs_property_returns_value(self):
        """Test logprobs property returns value when available."""
        twl = TokensWithLogprobs(tokens=[1], maybe_logprobs=[0.5])
        assert twl.logprobs == [0.5]

    def test_metadata_with_all_fields(self):
        """Test TokensWithLogprobs with fully populated metadata."""
        meta = SamplingMetadata(
            stop_reason="stop",
            prompt_logprobs=[-0.1, -0.2],
            topk_prompt_logprobs=[[(10, -0.05), (20, -0.1)], [(30, -0.15)]],
        )
        twl = TokensWithLogprobs(
            tokens=[100, 200],
            maybe_logprobs=[-0.5, -0.6],
            sampling_metadata=meta,
        )
        assert twl.stop_reason == "stop"
        assert twl.sampling_metadata.prompt_logprobs == [-0.1, -0.2]
        assert twl.sampling_metadata.topk_prompt_logprobs is not None
