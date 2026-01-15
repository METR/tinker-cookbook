"""Textual TUI application for viewing Tinker RL rollouts."""

import argparse
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from tinker_cookbook.rollout_viewer.channels import (
    CHANNEL_COLORS,
    extract_reasoning_from_harmony,
    parse_message_content,
)
from tinker_cookbook.rollout_viewer.data import Rollout, load_rollouts, watch_rollouts


class RolloutViewer(App[None]):
    """TUI for viewing Tinker RL rollouts."""

    TITLE = "Rollout Viewer"
    CSS = """
    #main {
        layout: horizontal;
    }
    #conversation {
        width: 70%;
        border: solid green;
        padding: 1;
    }
    #sidebar {
        width: 30%;
        border: solid blue;
        padding: 1;
    }
    #rollout-info {
        height: auto;
        border-bottom: solid $secondary;
        padding-bottom: 1;
        margin-bottom: 1;
    }
    #tokens-panel {
        height: auto;
        border-bottom: solid $secondary;
        padding-bottom: 1;
        margin-bottom: 1;
    }
    #reward-panel {
        height: auto;
        border-bottom: solid $secondary;
        padding-bottom: 1;
        margin-bottom: 1;
    }
    #scores-panel {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("left", "prev_rollout", "Prev Rollout"),
        Binding("right", "next_rollout", "Next Rollout"),
        Binding("h", "prev_rollout", "Prev Rollout", show=False),
        Binding("l", "next_rollout", "Next Rollout", show=False),
        Binding("j", "scroll_down", "Scroll Down", show=False),
        Binding("k", "scroll_up", "Scroll Up", show=False),
        Binding("r", "refresh", "Refresh"),
        Binding("w", "toggle_watch", "Watch Mode"),
    ]

    # Reactive state
    rollout_index: reactive[int] = reactive(0)
    watch_mode: reactive[bool] = reactive(False)

    def __init__(self, jsonl_path: Path, start_watching: bool = False) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        self.rollouts: list[Rollout] = []
        self.observer: Any = None
        self._start_watching = start_watching

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with ScrollableContainer(id="conversation"):
                yield Static("Loading...", id="turn-display")
            with Vertical(id="sidebar"):
                yield Static("", id="rollout-info")
                yield Static("", id="tokens-panel")
                yield Static("", id="reward-panel")
                yield Static("", id="scores-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.load_data()
        self.update_display()
        if self._start_watching:
            self.watch_mode = True
            self._start_file_watcher()

    def load_data(self) -> None:
        self.rollouts = load_rollouts(self.jsonl_path)
        if self.rollout_index >= len(self.rollouts):
            self.rollout_index = max(0, len(self.rollouts) - 1)

    def update_display(self) -> None:
        if not self.rollouts:
            self.query_one("#turn-display", Static).update("No rollouts loaded")
            self.query_one("#rollout-info", Static).update("No data")
            self.query_one("#tokens-panel", Static).update("")
            self.query_one("#reward-panel", Static).update("")
            self.query_one("#scores-panel", Static).update("")
            return

        rollout = self.rollouts[self.rollout_index]

        # Update rollout info
        info = Text()
        info.append(f"Rollout {self.rollout_index + 1}/{len(self.rollouts)}\n", style="bold")
        info.append(f"Step: {rollout.step}\n")
        # Color-code selection type
        selection_style = {
            "best": "bold green",
            "worst": "bold red",
            "random": "bold yellow",
            "only": "dim",
            "unknown": "dim",
        }.get(rollout.selection_type, "dim")
        info.append("Selection: ")
        info.append(f"{rollout.selection_type}\n", style=selection_style)
        info.append(f"Sample ID: {rollout.sample_id}\n")
        info.append(f"Turns: {len(rollout.conversation)}\n")
        info.append(f"Renderer: {rollout.renderer_name}\n")
        if self.watch_mode:
            info.append("[WATCHING]\n", style="bold yellow")
        info.append("\nh/l: rollouts  j/k: scroll", style="dim")
        self.query_one("#rollout-info", Static).update(info)

        # Update tokens panel - show token counts per turn
        tokens_text = Text()
        tokens_text.append("Tokens\n", style="bold underline")
        for i, message in enumerate(rollout.conversation):
            role = message.get("role", "unknown")

            # Use actual token counts if available, otherwise estimate
            if i < len(rollout.token_counts):
                tc = rollout.token_counts[i]
                content_tokens = tc.content_tokens
                reasoning_tokens = tc.reasoning_tokens
            else:
                # Fallback to estimation for old data
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

                # Handle unparsed Harmony format in content
                if not reasoning and isinstance(content, str) and "<|channel|>" in content:
                    extracted_reasoning, extracted_final = extract_reasoning_from_harmony(content)
                    reasoning_tokens = len(extracted_reasoning) // 4
                    content_tokens = len(extracted_final) // 4
                else:
                    content_tokens = len(content) // 4 if isinstance(content, str) else 0
                    reasoning_tokens = len(reasoning) // 4 if reasoning else 0

            tokens_text.append(f"{role}: ", style="bold")
            tokens_text.append(f"{content_tokens}", style="white")
            if reasoning_tokens > 0:
                tokens_text.append(" + ", style="dim")
                tokens_text.append(f"{reasoning_tokens}", style="cyan")
                tokens_text.append(" reasoning", style="dim cyan")
            tokens_text.append("\n")
        self.query_one("#tokens-panel", Static).update(tokens_text)

        # Update conversation display - show all turns
        text = Text()
        for i, message in enumerate(rollout.conversation):
            role = message.get("role", "unknown")
            parts = parse_message_content(message, rollout.renderer_name)

            # Role header
            text.append(f"{'─' * 40}\n", style="dim")
            text.append(f"[{role.upper()}]", style="bold underline")
            text.append(f" (turn {i + 1})\n\n", style="dim")

            # Message content with channel coloring
            for part in parts:
                color = CHANNEL_COLORS.get(part.channel, "white")
                if part.channel != "text":
                    text.append(f"[{part.channel}] ", style=f"dim {color}")
                text.append(part.content + "\n", style=color)
            text.append("\n")

        self.query_one("#turn-display", Static).update(text)

        # Update reward panel
        reward_text = Text()
        reward_text.append("Rewards\n", style="bold underline")
        reward_text.append(f"Total: {rollout.total_reward:.4f}\n\n", style="bold green")
        for name, value in rollout.individual_rewards.items():
            style = "green" if value > 0 else "dim"
            reward_text.append(f"{name}: {value:.4f}\n", style=style)
        self.query_one("#reward-panel", Static).update(reward_text)

        # Update scores panel
        scores_text = Text()
        scores_text.append("Scores\n", style="bold underline")
        for name, detail in rollout.scores.items():
            scores_text.append(f"\n{name}:\n", style="bold")
            scores_text.append(f"  value: {detail.get('value', 0):.4f}\n")
            answer = detail.get("answer")
            if answer:
                answer_preview = str(answer)[:50]
                if len(str(answer)) > 50:
                    answer_preview += "..."
                scores_text.append(f"  answer: {answer_preview}\n")
            metadata = detail.get("metadata", {})
            for k, v in metadata.items():
                scores_text.append(f"  {k}: {v}\n", style="dim")
        self.query_one("#scores-panel", Static).update(scores_text)

    def watch_rollout_index(self, old: int, new: int) -> None:
        self.update_display()

    def action_prev_rollout(self) -> None:
        if self.rollout_index > 0:
            self.rollout_index -= 1

    def action_next_rollout(self) -> None:
        if self.rollout_index < len(self.rollouts) - 1:
            self.rollout_index += 1

    def action_scroll_down(self) -> None:
        self.query_one("#conversation", ScrollableContainer).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one("#conversation", ScrollableContainer).scroll_up()

    def action_refresh(self) -> None:
        self.load_data()
        self.update_display()

    def action_toggle_watch(self) -> None:
        self.watch_mode = not self.watch_mode
        if self.watch_mode:
            self._start_file_watcher()
        else:
            self._stop_file_watcher()
        self.update_display()

    def _start_file_watcher(self) -> None:
        if self.observer is None:
            self.observer = watch_rollouts(self.jsonl_path, self._on_file_change)

    def _stop_file_watcher(self) -> None:
        if self.observer is not None:
            self.observer.stop()
            self.observer = None

    def _on_file_change(self) -> None:
        """Called when rollouts.jsonl is modified."""
        self.call_from_thread(self._handle_file_change)

    def _handle_file_change(self) -> None:
        """Handle file change on main thread."""
        self.load_data()
        # Jump to latest rollout
        if self.rollouts:
            self.rollout_index = len(self.rollouts) - 1
        self.update_display()

    def on_unmount(self) -> None:
        self._stop_file_watcher()


def main() -> None:
    parser = argparse.ArgumentParser(description="View Tinker RL rollouts")
    parser.add_argument("jsonl_path", help="Path to rollouts.jsonl")
    parser.add_argument("--watch", "-w", action="store_true", help="Auto-refresh on file changes")
    args = parser.parse_args()

    app = RolloutViewer(Path(args.jsonl_path), start_watching=args.watch)
    app.run()


if __name__ == "__main__":
    main()
