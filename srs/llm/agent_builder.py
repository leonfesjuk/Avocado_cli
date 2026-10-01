from typing import Any, Dict, List, Optional

from ..config.config_loader import (
    CONFIG,
    get_agent_config,
    get_experts_config,
    resolve_model_entry,
)
from .base import BaseEngine
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

    for role in expert_names:
        expert_cfg = experts_by_role.get(role)
        if expert_cfg is None:
            continue

        expert = _build_engine(
            expert_cfg.get("model", main_model_label),
            expert_cfg.get("name", role),
            expert_cfg.get("description", ""),
            expert_cfg.get("agent_type", agent_type),
        )
        if expert_cfg.get("allow_tools"):
            expert.register_tools(build_tools(expert_cfg.get("tools", [])))

        main_engine.register_expert(role, expert)

    return main_engine
