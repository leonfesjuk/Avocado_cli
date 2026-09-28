from typing import Any, Dict, List, Optional

from ..config.config_loader import (
    CONFIG,
    get_experts_config,
    resolve_model_entry,
)
from .base import BaseEngine
from .model_factory import create_model
from .smol_engine import SmolEngine
from .tool_registry import build_tools


def _supports_code_loop(provider: Dict[str, Any], model_cfg: Dict[str, Any]) -> bool:
    """Whether a model can drive smolagents' code-only loop.

    Groq's gpt-oss models emit native tool calls even when no tools are
    declared, which Groq rejects with HTTP 400 in the code-only loop. The
    tool-calling loop declares the built-in ``final_answer`` tool and works.
    """
    if provider.get("id") == "groq" and "gpt-oss" in model_cfg.get("name", ""):
        return False
    return True


def _build_engine(
    model_label: str,
    name: str,
    description: str,
    agent_type: str,
) -> SmolEngine:
    entry = resolve_model_entry(CONFIG, model_label)
    provider, model_cfg = entry["provider"], entry["model"]

    if agent_type == "code" and not _supports_code_loop(provider, model_cfg):
        agent_type = "tool"

    engine = SmolEngine(name=name, description=description, agent_type=agent_type)
    engine.set_model(create_model(provider, model_cfg))
    return engine


def build_agent(
    main_model_label: str,
    agent_name: str = "Avocado",
    agent_type: str = "code",
    expert_names: Optional[List[str]] = None,
) -> BaseEngine:
    """Build the main engine and optionally attach configured experts.

    Experts are created with the model referenced by their ``model`` field or,
    when absent, with the main model. Their ``allow_tools``/``tools`` config
    controls whether they receive tools.
    """
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
