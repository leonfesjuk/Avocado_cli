import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from smolagents import Tool

PLAN_SCHEMA_VERSION = 1
SEMANTIC_PLAN_KEYS = {"goal", "steps", "next_action", "notes"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_plan(goal: str = "") -> Dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA_VERSION,
        "status": "idle",
        "current_step": 0,
        "goal": goal,
        "steps": [],
        "next_action": "",
        "notes": "",
        "last_error": "",
        "updated_at": utc_now(),
    }


class UpdatePlanTool(Tool):
    """Let the model update the semantic parts of the persisted plan."""

    name = "update_plan"
    description = (
        "Update the persistent session plan. Pass a JSON object containing any "
        "of: goal (string), steps (list of step objects), next_action (string), "
        "notes (string). Status and step counters are managed by the engine."
    )
    inputs = {
        "plan_json": {
            "type": "string",
            "description": "JSON object with semantic plan fields to merge.",
        }
    }
    output_type = "string"

    def __init__(self, session: Any) -> None:
        self._session = session
        super().__init__()

    def forward(self, plan_json: str) -> str:
        try:
            update = json.loads(plan_json)
        except json.JSONDecodeError as error:
            return f"Plan update failed: invalid JSON: {error}"

        if not isinstance(update, dict):
            return "Plan update failed: expected a JSON object."

        unknown = sorted(set(update) - SEMANTIC_PLAN_KEYS)
        if unknown:
            return (
                "Plan update failed: unsupported field(s): "
                + ", ".join(unknown)
            )

        if "steps" in update and not isinstance(update["steps"], list):
            return "Plan update failed: 'steps' must be a list."

        plan = self._session.load_plan()
        for key, value in update.items():
            plan[key] = value
        plan["updated_at"] = utc_now()
        self._session.save_plan(plan)
        return "Plan updated."


def merge_plan(plan: Optional[Dict[str, Any]], **updates: Any) -> Dict[str, Any]:
    """Return a plan with engine-owned fields updated."""
    merged = default_plan()
    if isinstance(plan, dict):
        merged.update(plan)
    merged.update(updates)
    merged["updated_at"] = utc_now()
    return merged
