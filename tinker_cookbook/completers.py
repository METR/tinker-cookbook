"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import tinker

from tinker_cookbook import renderers

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]

# Matches tinker SDK's StopReason type exactly
StopReasonType: TypeAlias = Literal["length", "stop"]


@dataclass
class SamplingMetadata:
    """Metadata from the sampling process.

    Captures metadata returned by the tinker sampling client. Some fields
    are only populated if explicitly requested in SamplingParams.
    """

    stop_reason: StopReasonType
    """Reason why sampling stopped: 'length' (max_tokens) or 'stop' (stop sequence/EOS)."""

    prompt_logprobs: list[float | None] | None = None
    """Log probabilities for each prompt token. Only present if include_prompt_logprobs=True."""

    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None
    """Top-k log probabilities for each prompt token. Only present if topk_prompt_logprobs > 0."""


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
    sampling_metadata: SamplingMetadata | None = None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs

    @property
    def stop_reason(self) -> StopReasonType | None:
        """Convenience accessor for stop_reason from sampling_metadata."""
        if self.sampling_metadata is None:
            return None
        return self.sampling_metadata.stop_reason


class TokenCompleter:
    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a tinker.SamplingClient to sample actions.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        # Extract tokens and logprobs from the first (and only) sample
        sampled_sequence = sample_result.sequences[0]
        sampled_tokens = sampled_sequence.tokens
        sampled_logprobs = sampled_sequence.logprobs
        assert sampled_logprobs is not None

        # Capture sampling metadata (prompt_logprobs/topk_prompt_logprobs are None
        # unless explicitly requested in SamplingParams - infrastructure for future use)
        metadata = SamplingMetadata(
            stop_reason=sampled_sequence.stop_reason,
            prompt_logprobs=sample_result.prompt_logprobs,
            topk_prompt_logprobs=sample_result.topk_prompt_logprobs,
        )

        return TokensWithLogprobs(
            tokens=sampled_tokens,
            maybe_logprobs=sampled_logprobs,
            sampling_metadata=metadata,
        )


class TinkerMessageCompleter(MessageCompleter):
    """A completer that uses the actual model to generate responses."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
        temperature: float = 1.0,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)

        return {"role": "assistant", "content": parsed_message["content"]}
