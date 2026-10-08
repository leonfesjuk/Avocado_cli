import json
from typing import Any


def _plan_has_content(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    if plan.get("goal") or plan.get("steps") or plan.get("next_action") or plan.get("notes"):
        return True
    return plan.get("status") not in (None, "", "idle")


def build_instructions(session: Any, warn_uncompressed: bool = False) -> str:
    """Render persisted plan and long-term memory as agent instructions."""
    plan = session.load_plan()
    memory = session.load_memory().strip()
    raw_steps = session.raw_step_count()

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

    if not sections:
        return ""

    preamble = (
        "You are continuing a persisted Avocado session. The plan and memory "
        "below come from earlier turns and are authoritative context. Use them "
        "instead of asking the user to repeat work."
    )
    return preamble + "\n\n" + "\n\n".join(sections)

