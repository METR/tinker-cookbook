"""JSONL data loading and file watching for rollout viewer."""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer  # pyright: ignore[reportUnknownVariableType]

logger = logging.getLogger(__name__)


@dataclass
class TokenCount:
    """Token counts for a single message."""

    content_tokens: int
    reasoning_tokens: int


@dataclass
class Rollout:
    """Loaded rollout with parsed data."""

    index: int
    timestamp: str
    step: int
    selection_type: str  # "best", "worst", "random", "only"
    sample_id: str | int | None
    conversation: list[dict[str, Any]]
    token_counts: list[TokenCount]
    scores: dict[str, dict[str, Any]]
    individual_rewards: dict[str, float]
    total_reward: float
    renderer_name: str
    sample_info: dict[str, Any]
    stop_reason: str | None = None  # "stop" or "length"


def load_rollouts(path: Path) -> list[Rollout]:
    """Load all rollouts from JSONL file."""
    if not path.exists():
        return []

    rollouts: list[Rollout] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip partial/malformed lines (can happen if file is read while being written)
                logger.debug(f"Skipping malformed JSON at line {i + 1}")
                continue

            # Parse token counts
            token_counts = [
                TokenCount(
                    content_tokens=tc["content_tokens"],
                    reasoning_tokens=tc["reasoning_tokens"],
                )
                for tc in data["token_counts"]
            ]

            rollouts.append(
                Rollout(
                    index=i,
                    timestamp=data["timestamp"],
                    step=data["step"],
                    selection_type=data["selection_type"],
                    sample_id=data.get("sample_id"),  # Legitimately optional
                    conversation=data["conversation"],
                    token_counts=token_counts,
                    scores=data.get("scores", {}),  # Not always present
                    individual_rewards=data["individual_rewards"],
                    total_reward=data["total_reward"],
                    renderer_name=data["renderer_name"],
                    sample_info=data["sample_info"],
                    stop_reason=data.get("stop_reason"),
                )
            )

    return rollouts


class RolloutFileWatcher(FileSystemEventHandler):
    """Watch for changes to rollouts.jsonl."""

    def __init__(self, callback: Callable[[], None], file_name: str):
        super().__init__()
        self.callback = callback
        self.file_name = file_name
        self._last_modified = 0.0

    def on_modified(self, event: FileModifiedEvent) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if event.is_directory:
            return
        # Only trigger for our specific file
        if not str(event.src_path).endswith(self.file_name):
            return
        # Debounce rapid modifications
        now = time.time()
        if now - self._last_modified > 0.5:
            self._last_modified = now
            self.callback()


def watch_rollouts(path: Path, on_change: Callable[[], None]) -> Any:
    """Start watching rollouts file for changes.

    Args:
        path: Path to the rollouts.jsonl file
        on_change: Callback to invoke when file changes

    Returns:
        Observer that can be stopped with observer.stop()
    """
    handler = RolloutFileWatcher(on_change, path.name)
    observer = Observer()  # pyright: ignore[reportUnknownVariableType]
    observer.schedule(handler, str(path.parent), recursive=False)  # pyright: ignore[reportUnknownMemberType]
    observer.start()  # pyright: ignore[reportUnknownMemberType]
    return observer
