"""Channel detection and coloring for different renderers."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ChannelContent:
    """Parsed content with channel information."""

    channel: str  # "thinking", "text", "tool_call", "commentary", "tool_result"
    content: str


# Channel colors for Rich/Textual styling
CHANNEL_COLORS: dict[str, str] = {
    "thinking": "cyan",
    "text": "white",
    "tool_call": "yellow",
    "commentary": "magenta",
    "tool_result": "green",
}

# Compiled regex for Harmony format channel parsing
_HARMONY_PATTERN = re.compile(
    r"<\|channel\|>(\w+)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|call\|>|$)",
    re.DOTALL,
)


def extract_reasoning_from_harmony(content: str) -> tuple[str, str]:
    """Extract reasoning and final content from raw Harmony format.

    Returns:
        (reasoning_content, final_content) - text extracted from analysis and final channels
    """
    reasoning = ""
    final = ""

    for match in _HARMONY_PATTERN.finditer(content):
        channel = match.group(1)
        text = match.group(2)
        if channel == "analysis":
            reasoning += text
        elif channel == "final":
            final += text

    return reasoning, final


def _parse_harmony_content(content: str) -> list[ChannelContent]:
    """Parse Harmony format (GptOss) with channels.

    Harmony format uses:
    <|channel|>analysis<|message|>...<|end|>
    <|channel|>final<|message|>...<|return|>
    """
    parts: list[ChannelContent] = []

    for match in _HARMONY_PATTERN.finditer(content):
        channel = match.group(1)
        text = match.group(2)
        if channel == "analysis":
            parts.append(ChannelContent("thinking", text))
        elif channel == "final":
            parts.append(ChannelContent("text", text))
        elif channel == "commentary":
            parts.append(ChannelContent("commentary", text))
        else:
            parts.append(ChannelContent("text", text))

    return parts if parts else [ChannelContent("text", content)]


def _parse_qwen3_content(content: str) -> list[ChannelContent]:
    """Parse Qwen3 format with <think> blocks."""
    parts: list[ChannelContent] = []
    pos = 0

    # Pattern for <think>...</think> blocks
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    for match in pattern.finditer(content):
        # Text before think block
        if match.start() > pos:
            text = content[pos : match.start()]
            if text.strip():
                parts.append(ChannelContent("text", text))

        # Think block
        parts.append(ChannelContent("thinking", match.group(1)))
        pos = match.end()

    # Remaining text
    if pos < len(content):
        text = content[pos:]
        if text.strip():
            parts.append(ChannelContent("text", text))

    return parts if parts else [ChannelContent("text", content)]


def _parse_structured_content(content: list[dict[str, Any]]) -> list[ChannelContent]:
    """Parse structured content (list of ContentPart dicts)."""
    parts: list[ChannelContent] = []

    for part in content:
        part_type = part.get("type", "text")
        if part_type == "thinking":
            parts.append(ChannelContent("thinking", part["thinking"]))
        elif part_type == "text":
            parts.append(ChannelContent("text", part["text"]))
        elif part_type == "tool_call":
            tc = part["tool_call"]
            fn = tc.get("function", {})
            parts.append(
                ChannelContent(
                    "tool_call",
                    f"{fn.get('name', 'unknown')}({fn.get('arguments', '')})",
                )
            )
        else:
            parts.append(ChannelContent("text", str(part)))

    return parts if parts else [ChannelContent("text", str(content))]


def parse_message_content(
    message: dict[str, Any],
    renderer_name: str,
) -> list[ChannelContent]:
    """Parse message content into channel parts for coloring.

    Args:
        message: Message dict with role and content
        renderer_name: Name of the renderer class (e.g., "GptOssRenderer")

    Returns:
        List of ChannelContent with channel type and text
    """
    content = message.get("content", "")
    role = message.get("role", "")

    # Handle tool role
    if role == "tool":
        return [ChannelContent("tool_result", str(content))]

    # Handle structured content (list of parts)
    if isinstance(content, list):
        return _parse_structured_content(content)  # pyright: ignore[reportUnknownArgumentType]

    # Check for reasoning_content field (OpenAI-style normalized format)
    reasoning = message.get("reasoning_content")
    if reasoning:
        parts: list[ChannelContent] = [ChannelContent("thinking", reasoning)]
        if content:
            parts.append(ChannelContent("text", content))
        return parts

    # Parse string content based on renderer
    if "GptOss" in renderer_name:
        return _parse_harmony_content(content)
    elif "Qwen3" in renderer_name:
        return _parse_qwen3_content(content)
    else:
        # Generic fallback - just return as text
        return [ChannelContent("text", content)]
