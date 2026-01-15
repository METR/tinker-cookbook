"""Utilities for saving rollouts during Tinker RL training."""

import json
import logging
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from tinker_cookbook.renderers import Renderer

logger = logging.getLogger(__name__)

T = TypeVar("T")


def select_rollouts_to_save(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select best, worst, and one random rollout from a batch.

    Returns up to 3 unique rollouts with selection_type field:
    1. Worst (lowest total_reward)
    2. Best (highest total_reward)
    3. One random from the remaining
    """
    if len(rollouts) == 0:
        return []
    if len(rollouts) == 1:
        rollouts[0]["selection_type"] = "only"
        return rollouts

    sorted_rollouts = sorted(rollouts, key=lambda r: r["total_reward"])

    to_save: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    # Worst (lowest reward)
    worst_idx = 0
    worst = sorted_rollouts[worst_idx].copy()
    worst["selection_type"] = "worst"
    to_save.append(worst)
    used_indices.add(worst_idx)

    # Best (highest reward)
    best_idx = len(sorted_rollouts) - 1
    if best_idx not in used_indices:
        best = sorted_rollouts[best_idx].copy()
        best["selection_type"] = "best"
        to_save.append(best)
        used_indices.add(best_idx)

    # Random from the rest
    remaining_indices = [i for i in range(len(sorted_rollouts)) if i not in used_indices]
    if remaining_indices:
        rand_idx = random.choice(remaining_indices)
        rand_rollout = sorted_rollouts[rand_idx].copy()
        rand_rollout["selection_type"] = "random"
        to_save.append(rand_rollout)

    return to_save


def compute_message_tokens(
    message: dict[str, Any],
    tokenizer: Any,
) -> dict[str, int]:
    """Compute token counts for a message's content and reasoning."""
    content = message.get("content", "")
    reasoning = message.get("reasoning_content")

    content_tokens = 0
    if content:
        content_tokens = len(tokenizer.encode(content, add_special_tokens=False))  # pyright: ignore[reportUnknownMemberType]

    reasoning_tokens = 0
    if reasoning:
        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))  # pyright: ignore[reportUnknownMemberType]

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

        # Build rollout record
        record = record_builder(ctx, total_reward, rewards, step_counter, renderer_name)

        with lock:
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
