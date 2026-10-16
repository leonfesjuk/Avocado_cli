from typing import Any, Dict, List, Optional

from ..config.config_loader import (
    CONFIG,
    get_agent_config,
    get_experts_config,
    resolve_model_entry,
)
from .base import BaseEngine
from .expert_tools import ExpertTool, expert_tool_name
from .model_factory import create_model
from .smol_engine import SmolEngine
from .tool_registry import build_tools, list_tool_names

RESERVED_TOOL_NAMES = {"update_plan"}


def _supports_code_loop(model_cfg: Dict[str, Any]) -> bool:
    """Whether a model can drive smolagents' code-only loop.

    Some providers reject a code-only loop for specific models. The flag is
    configured per model with ``supports_code_loop``.
    """
    return bool(model_cfg.get("supports_code_loop", True))


def _build_engine(
    model_label: str,
    name: str,
    description: str,
    agent_type: str,
) -> SmolEngine:
    entry = resolve_model_entry(CONFIG, model_label)
    provider, model_cfg = entry["provider"], entry["model"]

    effective_type = model_cfg.get("agent_type", agent_type)
    if effective_type == "code" and not _supports_code_loop(model_cfg):
        effective_type = "tool"

    engine = SmolEngine(
        name=name,
        description=description,
        agent_type=effective_type,
    )
    engine.set_model(create_model(provider, model_cfg))
    return engine


def build_expert_tool(
    role: str,
    expert_cfg: Dict[str, Any],
    main_model_label: str,
    default_agent_type: str,
    expert_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ExpertTool:
    """Build a one-shot expert tool that reads runtime overrides on each call."""
    display_name = expert_cfg.get("name") or role
    description = expert_cfg.get("description", "")
    overrides = expert_overrides if expert_overrides is not None else {}

    def runner(task: str) -> str:
        override = overrides.get(role, {})
        allow_tools = override.get(
            "allow_tools", expert_cfg.get("allow_tools", False)
        )
        tool_names = override.get("tools", expert_cfg.get("tools", []))

        expert = _build_engine(
            expert_cfg.get("model", main_model_label),
            display_name,
            description,
            expert_cfg.get("agent_type", default_agent_type),
        )
        if allow_tools:
            expert.register_tools(build_tools(tool_names))
        return expert.run(task, reset=True)

    return ExpertTool(
        expert_tool_name(role),
        display_name,
        description,
        runner,
    )


def _validate_expert_tools(expert_cfg: Dict[str, Any]) -> None:
    if not expert_cfg.get("allow_tools"):
        return

    registered = set(list_tool_names())
    unknown = sorted(set(expert_cfg.get("tools", [])) - registered)
    if unknown:
        raise ValueError(
            f"Expert '{expert_cfg.get('role')}' references unknown tools: "
            + ", ".join(unknown)
        )


def build_agent(
    main_model_label: str,
    agent_name: Optional[str] = None,
    agent_type: Optional[str] = None,
    expert_names: Optional[List[str]] = None,
    expert_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BaseEngine:
    """Build the main engine and optionally attach configured experts.

    Experts are created with the model referenced by their ``model`` field or,
    when absent, with the main model. Their ``allow_tools``/``tools`` config
    controls whether they receive tools.
    """
    agent_config = get_agent_config(CONFIG)
    agent_name = agent_name if agent_name is not None else agent_config["agent_name"]
    agent_type = agent_type if agent_type is not None else agent_config["agent_type"]

    main_engine = _build_engine(main_model_label, agent_name, "", agent_type)

    if not expert_names:
        return main_engine

    experts_by_role = {
        expert["role"]: expert for expert in get_experts_config(CONFIG)
    }

    expert_tools = []
    seen_tool_names = set()
    for role in expert_names:
        expert_cfg = experts_by_role.get(role)
        if expert_cfg is None:
            raise KeyError(f"Expert '{role}' is not configured.")

        _validate_expert_tools(expert_cfg)
        tool_name = expert_tool_name(role)
        if tool_name in RESERVED_TOOL_NAMES:
            raise ValueError(f"Expert tool name '{tool_name}' is reserved.")
        if tool_name in seen_tool_names:
            raise ValueError(f"Duplicate expert tool name '{tool_name}'.")
        seen_tool_names.add(tool_name)

        tool = build_expert_tool(
            role,
            expert_cfg,
            main_model_label,
            agent_type,
            expert_overrides,
        )
        expert_tools.append(tool)
        main_engine.experts[role] = tool

    if expert_tools:
        main_engine.register_tools(expert_tools)

    return main_engine
