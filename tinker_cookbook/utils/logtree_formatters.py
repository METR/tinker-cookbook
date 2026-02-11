"""
HTML formatters for logtree.

This module provides formatter objects that encapsulate both HTML generation
and the CSS needed to style that HTML. Formatters implement the Formatter protocol
from logtree and can be logged using `logtree.log_formatter()`.
"""

import html
import json
from dataclasses import dataclass
from typing import Any, Sequence

from tinker_cookbook.renderers.base import Content, Message


def _render_content_html(content: Content) -> str:
    """Render message content as HTML with styled parts for thinking/tool calls."""
    if isinstance(content, str):
        return f'<span class="lt-text-part">{html.escape(content)}</span>'

    parts_html = []
    for part in content:
        if part["type"] == "text":
            parts_html.append(f'<span class="lt-text-part">{html.escape(part["text"])}</span>')
        elif part["type"] == "thinking":
            escaped = html.escape(part["thinking"])
            parts_html.append(
                f'<details class="lt-thinking-part">'
                f"<summary>💭 Thinking</summary>"
                f"<pre>{escaped}</pre>"
                f"</details>"
            )
        elif part["type"] == "tool_call":
            tc = part["tool_call"]
            name = html.escape(tc.function.name)
            args = html.escape(tc.function.arguments)
            parts_html.append(
                f'<div class="lt-tool-call-part">'
                f'<span class="lt-tool-call-label">🔧 Tool Call:</span> '
                f"<code>{name}({args})</code>"
                f"</div>"
            )
        elif part["type"] == "unparsed_tool_call":
            raw = html.escape(part["raw_text"])
            error = html.escape(part["error"])
            parts_html.append(
                f'<div class="lt-unparsed-tool-call-part">'
                f'<span class="lt-tool-call-label">⚠️ Unparsed Tool Call:</span> '
                f"<code>{raw}</code>"
                f'<div class="lt-error">{error}</div>'
                f"</div>"
            )
        elif part["type"] == "image":
            parts_html.append('<span class="lt-image-part">🖼️ [Image]</span>')
        else:
            raise ValueError(f"Unknown content part type: {part['type']}")
    return "\n".join(parts_html)


def _render_top_level_tool_calls(msg: dict[str, Any]) -> str:
    """Render tool_calls / unparsed_tool_calls stored as top-level message fields."""
    parts: list[str] = []
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        name = html.escape(fn.get("name", "unknown"))
        raw_args = fn.get("arguments", "")
        try:
            args_str = html.escape(json.dumps(json.loads(raw_args), indent=2))
        except (json.JSONDecodeError, TypeError):
            args_str = html.escape(raw_args)
        parts.append(
            f'<div class="lt-tool-call-part">'
            f'<span class="lt-tool-call-label">🔧 Tool Call:</span> '
            f"<code>{name}({args_str})</code>"
            f"</div>"
        )
    for utc in msg.get("unparsed_tool_calls", []):
        raw = html.escape(utc.get("raw_text", str(utc)))
        error = html.escape(utc.get("error", ""))
        parts.append(
            f'<div class="lt-unparsed-tool-call-part">'
            f'<span class="lt-tool-call-label">⚠️ Unparsed Tool Call:</span> '
            f"<code>{raw}</code>"
            f'<div class="lt-error">{error}</div>'
            f"</div>"
        )
    return "\n".join(parts)


def _render_tool_meta(msg: dict[str, Any]) -> str:
    """Render function name and tool_call_id for tool-result messages."""
    label_parts: list[str] = []
    if name := msg.get("name"):
        label_parts.append(html.escape(name))
    if tool_call_id := msg.get("tool_call_id"):
        label_parts.append(f"id={html.escape(tool_call_id)}")
    if not label_parts:
        return ""
    return f'<span class="lt-tool-meta">{" ".join(label_parts)}</span>'


@dataclass
class ConversationFormatter:
    """
    Formatter for conversation messages.

    Renders a list of messages as a styled conversation with role-based coloring.
    Supports structured content with thinking parts, tool calls, and text.
    """

    messages: Sequence[Message]
    """List of messages, each with 'role' and 'content' keys."""

    def to_html(self) -> str:
        """Generate HTML for the conversation."""
        parts = ['<div class="lt-conversation">']
        for msg in self.messages:
            role = html.escape(msg["role"])
            content_html = _render_content_html(msg["content"])
            extra_html = _render_top_level_tool_calls(msg)
            tool_meta_html = _render_tool_meta(msg) if role == "tool" else ""
            parts.append(f'  <div class="lt-message lt-message-{role}">')
            parts.append(f'    <span class="lt-message-role">{role}:</span>')
            if tool_meta_html:
                parts.append(f"    {tool_meta_html}")
            parts.append(f'    <div class="lt-message-content">{content_html}{extra_html}</div>')
            parts.append("  </div>")
        parts.append("</div>")
        return "\n".join(parts)

    def get_css(self) -> str:
        """Get CSS for conversation styling."""
        return CONVERSATION_CSS


# CSS for conversation formatting
CONVERSATION_CSS = """
/* Conversation formatting */
.lt-conversation {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.lt-message {
    padding: 0.75rem;
    border-radius: 6px;
    border-left: 3px solid var(--lt-accent, #2563eb);
    background: var(--lt-bg, #f9fafb);
    line-height: 1.5;
}

.lt-message-role {
    font-weight: 600;
    color: var(--lt-accent, #2563eb);
    display: inline-block;
    min-width: 80px;
}

.lt-message-content {
    white-space: pre-wrap;
    word-wrap: break-word;
}

.lt-message-user {
    background: #e3f2fd;
    border-left-color: #1976d2;
}

.lt-message-user .lt-message-role {
    color: #1565c0;
}

.lt-message-assistant {
    background: #f3e5f5;
    border-left-color: #7b1fa2;
}

.lt-message-assistant .lt-message-role {
    color: #6a1b9a;
}

.lt-message-system {
    background: #fff3e0;
    border-left-color: #f57c00;
}

.lt-message-system .lt-message-role {
    color: #e65100;
}

.lt-message-tool {
    background: #e8f5e9;
    border-left-color: #388e3c;
}

.lt-message-tool .lt-message-role {
    color: #2e7d32;
}

.lt-tool-meta {
    display: inline-block;
    margin-left: 0.5rem;
    padding: 0.1rem 0.4rem;
    background: #d1fae5;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #065f46;
}

/* Content part styling */
.lt-text-part {
    display: block;
}

.lt-thinking-part {
    margin: 0.5rem 0;
    padding: 0.5rem;
    background: #fef3c7;
    border: 1px solid #fbbf24;
    border-radius: 4px;
}

.lt-thinking-part summary {
    cursor: pointer;
    font-weight: 500;
    color: #92400e;
}

.lt-thinking-part pre {
    margin: 0.5rem 0 0 0;
    padding: 0.5rem;
    background: #fffbeb;
    border-radius: 4px;
    font-size: 0.875rem;
    overflow-x: auto;
    white-space: pre-wrap;
}

.lt-tool-call-part {
    margin: 0.5rem 0;
    padding: 0.5rem;
    background: #dbeafe;
    border: 1px solid #3b82f6;
    border-radius: 4px;
}

.lt-tool-call-label {
    font-weight: 500;
    color: #1e40af;
}

.lt-tool-call-part code {
    display: block;
    margin-top: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: #eff6ff;
    border-radius: 2px;
    font-size: 0.875rem;
    overflow-x: auto;
}

.lt-unparsed-tool-call-part {
    margin: 0.5rem 0;
    padding: 0.5rem;
    background: #fee2e2;
    border: 1px solid #ef4444;
    border-radius: 4px;
}

.lt-unparsed-tool-call-part .lt-tool-call-label {
    color: #991b1b;
}

.lt-unparsed-tool-call-part code {
    display: block;
    margin-top: 0.25rem;
    padding: 0.25rem 0.5rem;
    background: #fef2f2;
    border-radius: 2px;
    font-size: 0.875rem;
    overflow-x: auto;
}

.lt-error {
    margin-top: 0.25rem;
    font-size: 0.875rem;
    color: #dc2626;
}

.lt-image-part {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background: #e0e7ff;
    border-radius: 4px;
    color: #3730a3;
}
"""
