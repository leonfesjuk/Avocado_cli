import json
from typing import Any, Dict, List, Optional

from smolagents import ChatMessage, MessageRole

from .token_counter import estimate_tokens, limit_text

DEFAULT_SUMMARY_PROMPT = """You are compressing the working memory of an AI coding session.

Write a compact markdown memory that lets the assistant continue the session
without reading the raw transcript. Preserve:
- the user's goal and explicit constraints;
- important decisions and rejected alternatives;
- files, symbols and commands that were changed or inspected;
- verified results, errors, and unresolved problems;
- the next concrete action.

The current plan and the previous long-term memory are authoritative. Merge
them with the new events, remove duplicates, and do not invent facts. Return
only the markdown memory, with no preamble and no code fences.
Aim for at most {max_tokens} tokens.

## Current plan
{plan}

## Previous long-term memory
{memory}

## New session events
{transcript}
"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _clip(value: Any, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, default=str
    )
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n[...{len(text) - limit} chars clipped]"


def format_steps(
    steps: List[Dict[str, Any]],
    max_tokens: Optional[int] = None,
) -> str:
    blocks = []
    for record in steps:
        kind = record.get("kind", "step")
        number = record.get("step")
        header = f"### {kind}"
        if number is not None:
            header += f" {number}"
        lines = [header]

        for key in ("task", "model_output", "plan", "observations", "output", "error"):
            value = record.get(key)
            if value:
                lines.append(f"{key}: {_clip(value)}")
        tool_calls = record.get("tool_calls")
        if tool_calls:
            lines.append(f"tool_calls: {_clip(tool_calls)}")
        blocks.append("\n".join(lines))

    if not max_tokens or max_tokens <= 0 or not blocks:
        return "\n\n".join(blocks)

    selected = []
    used = 0
    for block in reversed(blocks):
        cost = estimate_tokens(block) + 1
        if selected and used + cost > max_tokens:
            break
        if not selected and cost > max_tokens:
            selected.append(limit_text(block, max_tokens))
            break
        selected.append(block)
        used += cost

    selected.reverse()
    if len(selected) < len(blocks):
        selected.insert(
            0,
            "[...older events omitted to fit the transcript budget...]",
        )
    return "\n\n".join(selected)


class MemoryCompressor:
    """Fold a session's raw step journal into ``memory.md``."""

    def __init__(
        self,
        model: Any,
        max_summary_tokens: int = 1200,
        max_transcript_tokens: int = 6000,
        prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        self.model = model
        self.max_summary_tokens = max_summary_tokens
        self.max_transcript_tokens = max_transcript_tokens
        self.prompt_template = prompt_template

    def compress(self, session: Any) -> str:
        steps = session.read_steps()
        previous_memory = session.load_memory().strip()

        if not steps:
            return previous_memory

        plan = session.load_plan()
        prompt = self.prompt_template.format(
            max_tokens=self.max_summary_tokens,
            plan=json.dumps(plan, ensure_ascii=False, indent=2),
            memory=previous_memory or "(none)",
            transcript=format_steps(steps, self.max_transcript_tokens),
        )
        response = self.model.generate(
            [ChatMessage(role=MessageRole.USER, content=prompt)]
        )
        summary = _content_to_text(getattr(response, "content", response)).strip()
        if not summary:
            raise RuntimeError("Compression model returned an empty summary.")

        summary = limit_text(summary, self.max_summary_tokens)
        archive_path = session.archive_current_run()
        metadata = {
            "kind": "compressed",
            "compacted_steps": len(steps),
            "estimated_tokens": estimate_tokens(summary),
            "max_summary_tokens": self.max_summary_tokens,
            "max_transcript_tokens": self.max_transcript_tokens,
            "model": getattr(self.model, "model_id", ""),
        }
        if archive_path:
            metadata["archive_path"] = str(archive_path)
        session.save_memory(summary + "\n", metadata)
        return summary
