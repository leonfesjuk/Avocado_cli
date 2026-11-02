import json
from typing import Any, List, Optional

from ..config.config_loader import CONFIG, get_agent_config

EXPERT_DELEGATION_PROMPT = """## Delegating to experts

You can delegate self-contained subtasks to the expert tools available in this
session. Experts are stateless: they cannot see this conversation, the session
plan, or long-term memory. When calling an expert:
- Provide only the context that subtask needs: the precise question, relevant
  paths or snippets, constraints, and the expected output.
- Do not dump the whole conversation, the plan, or unrelated memory.
- Split large problems into focused calls instead of one broad request.
- Verify and integrate the expert's answer yourself; you own the final result.
- Do the work yourself when it is simple; delegate only when the expert's
  specialty adds value."""


def _plan_has_content(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    if plan.get("goal") or plan.get("steps") or plan.get("next_action") or plan.get("notes"):
        return True
    return plan.get("status") not in (None, "", "idle")


def build_instructions(
    session: Any,
    warn_uncompressed: bool = False,
    experts: Optional[List[str]] = None,
) -> str:
    """Render persisted context and expert guidance as agent instructions."""
    plan = session.load_plan()
    memory = session.load_memory().strip()
    raw_steps = session.raw_step_count()

    parts = []
    base_prompt = get_agent_config(CONFIG).get("system_prompt", "").strip()
    if base_prompt:
        parts.append(base_prompt)
    if experts:
        parts.append(EXPERT_DELEGATION_PROMPT)

    sections = []
    if _plan_has_content(plan):
        sections.append(
            "## Session plan\n"
            + json.dumps(plan, ensure_ascii=False, indent=2)
        )
    if memory:
        sections.append("## Long-term memory\n" + memory)
    if warn_uncompressed and raw_steps and not memory:
        sections.append(
            "## Uncompressed history\n"
            f"{raw_steps} session events are stored on disk but are not loaded. "
            "Run /compress to fold them into long-term memory."
        )

    if sections:
        parts.append(
            "You are continuing a persisted Avocado session. The plan and memory "
            "below come from earlier turns and are authoritative context. Use them "
            "instead of asking the user to repeat work."
        )
        parts.extend(sections)

    return "\n\n".join(parts)

