import re
from typing import Any, Dict, List, Optional

from ..config.config_loader import (
    CONFIG,
    get_agent_config,
    get_experts_config,
    resolve_model_entry,
)
from .base import BaseEngine
from .expert_tools import ExpertTool
from .model_factory import create_model
from .smol_engine import SmolEngine
from .tool_registry import build_tools


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


def build_agent(
    main_model_label: str,
    agent_name: Optional[str] = None,
    agent_type: Optional[str] = None,
    expert_names: Optional[List[str]] = None,
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
    for role in expert_names:
        expert_cfg = experts_by_role.get(role)
        if expert_cfg is None:
            continue

        tool_name = re.sub(r"\W", "_", role)
        if not tool_name or tool_name[0].isdigit():
            tool_name = f"expert_{tool_name}"

        def runner(
            task: str,
            cfg: Dict[str, Any] = expert_cfg,
            role_name: str = role,
        ) -> str:
            expert = _build_engine(
                cfg.get("model", main_model_label),
                cfg.get("name", role_name),
                cfg.get("description", ""),
                cfg.get("agent_type", agent_type),
            )
            if cfg.get("allow_tools"):
                expert.register_tools(build_tools(cfg.get("tools", [])))
            return expert.run(task, reset=True)

        description = expert_cfg.get("description", "")
        tool = ExpertTool(
            role=tool_name,
            name=expert_cfg.get("name", role),
            description=description,
            runner=runner,
        )
        expert_tools.append(tool)
        main_engine.experts[role] = tool

    if expert_tools:
        main_engine.register_tools(expert_tools)

    return main_engine
