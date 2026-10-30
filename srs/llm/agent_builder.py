from typing import Any, Dict, List, Optional

from smolagents.monitoring import AgentLogger, LogLevel

from ..config.config_loader import (
    CONFIG,
    get_agent_config,
    get_experts_config,
    get_model_capabilities,
    resolve_model_entry,
)
from .base import BaseEngine
from .chat_engine import ChatEngine
from .expert_tools import ExpertTool, expert_tool_name, last_expert_calls
from .logger import AvocadoAgentLogger
from .model_factory import create_model
from .smol_engine import SmolEngine
from .tool_registry import build_tools, list_tool_names

RESERVED_TOOL_NAMES = {"update_plan"}


def _effective_agent_type(model_cfg: Dict[str, Any], requested: str) -> str:
    """Pick the best supported loop for a model and requested agent type."""
    if requested == "chat":
        return "chat"

    capabilities = get_model_capabilities(model_cfg)
    if requested == "code" and capabilities["supports_code_loop"]:
        return "code"
    if requested == "tool" and capabilities["supports_tool_calling"]:
        return "tool"
    if capabilities["supports_code_loop"]:
        return "code"
    if capabilities["supports_tool_calling"]:
        return "tool"
    return "chat"


def _build_engine(
    model_label: str,
    name: str,
    description: str,
    agent_type: str,
    logger: Optional[AgentLogger] = None,
) -> BaseEngine:
    entry = resolve_model_entry(CONFIG, model_label)
    provider, model_cfg = entry["provider"], entry["model"]

    effective_type = _effective_agent_type(model_cfg, agent_type)
    if effective_type == "chat":
        engine: BaseEngine = ChatEngine(name=name, description=description)
    else:
        engine = SmolEngine(
            name=name,
            description=description,
            agent_type=effective_type,
            logger=logger,
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

        requested_type = expert_cfg.get("agent_type", default_agent_type)
        if not allow_tools:
            requested_type = "chat"
        elif requested_type == "chat":
            requested_type = "tool"

        expert = _build_engine(
            expert_cfg.get("model", main_model_label),
            display_name,
            description,
            requested_type,
            logger=AgentLogger(level=LogLevel.OFF),
        )
        if allow_tools:
            if not getattr(expert, "supports_tools", True):
                raise ValueError(
                    f"Expert '{role}' model does not support tools. "
                    "Choose a tool-capable model or disable expert tools."
                )
            expert.register_tools(build_tools(tool_names))

        system_prompt = expert_cfg.get("system_prompt")
        if not system_prompt:
            system_prompt = f"You are {display_name}. {description}".strip()
        if hasattr(expert, "set_instructions"):
            expert.set_instructions(system_prompt)

        try:
            response = expert.run(task, reset=True)
            last_expert_calls.append(
                {
                    "role": role,
                    "task": task,
                    "response": response,
                    "error": None,
                }
            )
            return response
        except Exception as error:
            last_expert_calls.append(
                {
                    "role": role,
                    "task": task,
                    "response": None,
                    "error": str(error),
                }
            )
            return (
                f"Expert {display_name} unavailable: {error}. "
                "I will answer without the expert."
            )

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
    tools_enabled: Optional[bool] = None,
) -> BaseEngine:
    """Build the main engine and optionally attach configured experts.

    ``tools_enabled`` controls whether the main model gets an agent loop at
    all. When it is false the result is a stateless :class:`ChatEngine`.
    """
    agent_config = get_agent_config(CONFIG)
    agent_name = agent_name if agent_name is not None else agent_config["agent_name"]
    agent_type = agent_type if agent_type is not None else agent_config["agent_type"]
    if tools_enabled is None:
        tools_enabled = bool(expert_names)

    if not tools_enabled:
        return _build_engine(
            main_model_label,
            agent_name,
            "",
            "chat",
        )

    main_engine = _build_engine(
        main_model_label,
        agent_name,
        "",
        agent_type,
        logger=AvocadoAgentLogger(),
    )
    if not main_engine.supports_tools:
        raise ValueError(
            f"Model '{main_model_label}' does not support tools. "
            "Switch to a tool-capable model before enabling tools or experts."
        )

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
