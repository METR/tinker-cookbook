"""Utilities for saving rollouts during Tinker RL training."""

import json
import logging
import math
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from tinker_cookbook.renderers import Renderer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Tokenizer(Protocol):
    """Protocol for tokenizers used in rollout saving."""

    def encode(
        self, text: str, *args: Any, add_special_tokens: bool = ..., **kwargs: Any
    ) -> list[int]: ...


def _sort_key_with_nan(r: dict[str, Any]) -> tuple[bool, float]:
    """Sort key that puts NaN values at the end."""
    reward = r["total_reward"]
    return (math.isnan(reward), reward)


def select_rollouts_to_save(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select best, worst, and one random rollout from a batch.

    Returns up to 3 unique rollouts with selection_type field:
    1. Worst (lowest total_reward)
    2. Best (highest total_reward)
    3. One random from the remaining

    NaN rewards are sorted to the end and excluded from best/worst selection.
    """
    if len(rollouts) == 0:
        return []
    if len(rollouts) == 1:
        only = rollouts[0].copy()
        only["selection_type"] = "only"
        return [only]

    # Sort with NaN values at the end
    sorted_rollouts = sorted(rollouts, key=_sort_key_with_nan)

    # Warn if any NaN rewards are present
    nan_count = sum(1 for r in rollouts if math.isnan(r["total_reward"]))
    if nan_count > 0:
        logger.warning(f"Found {nan_count} rollouts with NaN rewards out of {len(rollouts)}")

    worst = sorted_rollouts[0].copy()
    worst["selection_type"] = "worst"

    best = sorted_rollouts[-1].copy()
    best["selection_type"] = "best"

    to_save = [worst, best]

    # Random from middle (indices 1 to len-2)
    if len(sorted_rollouts) > 2:
        rand_idx = random.randint(1, len(sorted_rollouts) - 2)
        rand_rollout = sorted_rollouts[rand_idx].copy()
        rand_rollout["selection_type"] = "random"
        to_save.append(rand_rollout)

    return to_save


def compute_message_tokens(
    message: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, int]:
    """Compute token counts for a message's content and reasoning."""
    content = message.get("content", "")
    reasoning = message.get("reasoning_content")

    content_tokens = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0
    reasoning_tokens = (
        len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0
    )

    return {
        "content_tokens": content_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def with_rollout_saving(
    inner_fn: Callable[[T], tuple[float, dict[str, float]]],
    output_path: Path,
    renderer: Renderer,
    samples_per_batch: int,
    save_every: int = 10,
    build_record: Callable[[T, float, dict[str, float], int, str], dict[str, Any]] | None = None,
) -> Callable[[T], tuple[float, dict[str, float]]]:
    """Wrap a reward function to save rollouts periodically.

    Saves 3 rollouts (best, worst, random) every `save_every` steps.

    Args:
        inner_fn: The original reward function to wrap
        output_path: Path to the rollouts.jsonl file
        renderer: Tinker Renderer instance (used for renderer_name and tokenizer)
        samples_per_batch: Number of samples per batch (batch_size * group_size)
        save_every: Save rollouts every N steps (default: 10)
        build_record: Optional function to build a custom rollout record.
            Signature: (ctx, total_reward, rewards, step, renderer_name) -> dict
            If not provided, a default record builder is used that expects ctx to have
            'conversation', 'sample_info', and optionally 'scores' attributes/keys.

    Returns:
        Wrapped function that periodically saves selected rollouts to JSONL
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer_name = renderer.__class__.__name__
    tokenizer = renderer.tokenizer

    # Thread-safe state
    lock = threading.Lock()
    batch_buffer: list[dict[str, Any]] = []
    step_counter = 0

    def default_build_record(
        ctx: Any,
        total_reward: float,
        rewards: dict[str, float],
        step: int,
        rname: str,
    ) -> dict[str, Any]:
        """Default record builder for contexts with conversation and sample_info."""
        # Try to get conversation and sample_info from ctx
        conversation: list[dict[str, Any]] = []
        sample_info: dict[str, Any] = {}

        if hasattr(ctx, "conversation"):
            conversation = list(ctx.conversation)
        elif isinstance(ctx, dict) and "conversation" in ctx:
            conversation = list(ctx["conversation"])

        if hasattr(ctx, "sample_info"):
            sample_info = dict(getattr(ctx, "sample_info"))
        elif isinstance(ctx, dict) and "sample_info" in ctx:
            sample_info = dict(ctx["sample_info"])

        # Compute token counts
        token_counts = [compute_message_tokens(msg, tokenizer) for msg in conversation]

        return {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "sample_id": sample_info.get("inspect_sample_id"),
            "conversation": conversation,
            "token_counts": token_counts,
            "sample_info": sample_info,
            "individual_rewards": rewards,
            "total_reward": total_reward,
            "renderer_name": rname,
        }

    record_builder = build_record if build_record is not None else default_build_record

    def wrapper(ctx: T) -> tuple[float, dict[str, float]]:
        nonlocal batch_buffer, step_counter

        # Call inner function first to get rewards
        total_reward, rewards = inner_fn(ctx)

        with lock:
            # Build rollout record inside lock to get consistent step_counter
            record = record_builder(ctx, total_reward, rewards, step_counter, renderer_name)
            batch_buffer.append(record)

            # Check if batch is complete
            if len(batch_buffer) >= samples_per_batch:
                step_counter += 1

                # Save every N steps
                if step_counter % save_every == 0:
                    selected = select_rollouts_to_save(batch_buffer)
                    with open(output_path, "a") as f:
                        for r in selected:
                            f.write(json.dumps(r) + "\n")
                    logger.info(f"Saved {len(selected)} rollouts at step {step_counter}")

                batch_buffer = []

        return total_reward, rewards

    return wrapper
