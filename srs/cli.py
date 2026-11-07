import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .sessions.manager import SessionManager
from .sessions.store import SessionStore
from .ui import (
    CONSOLE,
    render_answer,
    render_banner,
    render_expert_calls,
    render_status,
)

HELP_TEXT = """Commands:
  /help                 show this help
  /sessions             list saved sessions
  /new [name]           create and switch to a new session
  /switch <id|name>     compress the current session, then switch
  /rename <name>        rename the current session
  /delete <id|name>     delete a non-active session
  /plan                 show the current plan
  /memory               show memory paths and preview
  /model [list]         list models and their capabilities
  /model <label>        switch the session model
  /tools [list]         list registered and active tools
  /tools add <name|*>   attach one tool or every built-in tool
  /tools remove <name>  detach a tool from this session
  /tools clear          detach every tool from this session
  /experts [list]       list experts, expertise and their tools
  /experts enable <role|*>   attach one or all experts
  /experts disable <role|*>  detach one or all experts
  /experts tools <role> [tool ...|none]
                        set the tools an expert may use
  /compress             fold the raw step journal into memory.md
  /compress-model [list]  list models used for memory compression
  /compress-model <label> set the model used for memory compression
  /compress-model default follow the session model again
  /reset [--hard]       clear current history; --hard also clears memory
  /context              show estimated context usage and limits
  /quit                 save and exit
  /exit                 alias for /quit

Any other input is sent to the main agent.
"""


def _fmt_tokens(value: int) -> str:
    return f"{value:,}"


def _print_context(manager: SessionManager, label: str = "Context") -> None:
    stats = manager.context_usage()
    if stats["limit"]:
        budget = (
            f"{_fmt_tokens(stats['used'])} / {_fmt_tokens(stats['limit'])}"
            f" tokens ({stats['percent']:.1f}%)"
        )
    else:
        budget = f"{_fmt_tokens(stats['used'])} tokens (limit unknown)"
    render_status(
        f"{label} (est.): {budget} | history {_fmt_tokens(stats['history_tokens'])}"
        f" | memory {_fmt_tokens(stats['memory_tokens'])}"
        f" | raw {_fmt_tokens(stats['raw_history_tokens'])} ({stats['raw_steps']} events)"
    )


def _print_context_details(manager: SessionManager) -> None:
    stats = manager.context_usage()
    if stats["limit"]:
        budget = (
            f"{_fmt_tokens(stats['used'])} / {_fmt_tokens(stats['limit'])}"
            f" tokens ({stats['percent']:.1f}%)"
        )
    else:
        budget = f"{_fmt_tokens(stats['used'])} tokens (limit unknown)"
    render_status(f"Model: {stats['model']}")
    render_status(
        "Compression model: "
        + (manager.compression_model_label or stats["model"])
    )
    render_status(f"Context (est.): {budget}")
    render_status(f"  system+instructions: {_fmt_tokens(stats['prompt_tokens'])} tokens")
    render_status(f"  in-memory history:   {_fmt_tokens(stats['history_tokens'])} tokens")
    render_status(f"  long-term memory:    {_fmt_tokens(stats['memory_tokens'])} tokens")
    render_status(f"  plan:                {_fmt_tokens(stats['plan_tokens'])} tokens")
    render_status(
        f"  raw journal:         {_fmt_tokens(stats['raw_history_tokens'])} tokens "
        f"({stats['raw_steps']} events)"
    )
    if stats["last_input_tokens"]:
        render_status(
            f"  last request input:  {_fmt_tokens(stats['last_input_tokens'])} tokens"
        )


def _print_compression(manager: SessionManager) -> None:
    stats = manager.last_compression
    if not stats:
        return
    before = stats["context_before"]
    after = stats["context_after"]
    delta = stats["context_delta_tokens"]
    pct = stats["context_delta_percent"]
    if delta >= 0:
        word = "saved"
        sign = ""
    else:
        word = "increased"
        sign = "+"
    render_status(
        f"Compression: context {_fmt_tokens(before['used'])} → {_fmt_tokens(after['used'])}"
        f" ({word} {sign}{_fmt_tokens(abs(delta))}, {abs(pct):.1f}%)"
    )
    render_status(
        f"  history:   {_fmt_tokens(before['history_tokens'])} → {_fmt_tokens(after['history_tokens'])} tokens"
    )
    render_status(
        f"  memory:    {_fmt_tokens(before['memory_tokens'])} → {_fmt_tokens(after['memory_tokens'])} tokens"
    )
    render_status(
        f"  raw journal: {_fmt_tokens(before['raw_history_tokens'])} →"
        f" {_fmt_tokens(after['raw_history_tokens'])} tokens"
    )
    if stats["journal_saved_tokens"]:
        render_status(
            f"  journal source: {_fmt_tokens(stats['source_tokens'])} →"
            f" {_fmt_tokens(stats['summary_tokens'])} tokens"
            f" (saved {_fmt_tokens(stats['journal_saved_tokens'])}, {stats['journal_saved_percent']:.1f}%)"
        )
    _print_context(manager, label="Context now")


def _print_compression_stats(stats: Dict[str, Any]) -> None:
    before = stats["context_before"]
    after = stats["context_after"]
    delta = stats["context_delta_tokens"]
    pct = stats["context_delta_percent"]
    if delta >= 0:
        word = "saved"
        sign = ""
    else:
        word = "increased"
        sign = "+"
    render_status(
        f"Compression: context {_fmt_tokens(before['used'])} → {_fmt_tokens(after['used'])}"
        f" ({word} {sign}{_fmt_tokens(abs(delta))}, {abs(pct):.1f}%)"
    )
    render_status(
        f"  history:   {_fmt_tokens(before['history_tokens'])} → {_fmt_tokens(after['history_tokens'])} tokens"
    )
    render_status(
        f"  memory:    {_fmt_tokens(before['memory_tokens'])} → {_fmt_tokens(after['memory_tokens'])} tokens"
    )
    render_status(
        f"  raw journal: {_fmt_tokens(before['raw_history_tokens'])} →"
        f" {_fmt_tokens(after['raw_history_tokens'])} tokens"
    )


def _print_sessions(manager: SessionManager) -> None:
    sessions = manager.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    for item in sessions:
        marker = (
            "*"
            if manager.session and item["id"] == manager.session.session_id
            else " "
        )
        print(
            f"{marker} {item['id']}  {item['name']}  "
            f"steps={item['steps']}  memory={item['memory_chars']}ch  "
            f"updated={item['updated_at']}"
        )


def _handle_command(manager: SessionManager, line: str) -> bool:
    try:
        parts = shlex.split(line)
    except ValueError as error:
        print(f"Invalid command: {error}")
        return True
    if not parts:
        return True

    command = parts[0]
    args = parts[1:]

    if command in {"/quit", "/exit"}:
        return False
    if command == "/help":
        print(HELP_TEXT, end="")
    elif command == "/sessions":
        _print_sessions(manager)
    elif command == "/new":
        session = manager.new(name=" ".join(args) or None)
        print(f"Created session {session.name} ({session.session_id}).")
        if manager.last_warning:
            print(f"Warning: {manager.last_warning}")
        _print_context(manager)
    elif command == "/switch":
        if not args:
            print("Usage: /switch <id|name>")
        else:
            result = manager.switch(" ".join(args))
            session = result["session"]
            print(f"Switched to {session.name} ({session.session_id}).")
            if result.get("compressed"):
                print("Previous session was compressed.")
                if result.get("compression"):
                    _print_compression_from_stats(result["compression"])
            if result["warning"]:
                print(f"Warning: {result['warning']}")
            if result.get("compression"):
                _print_compression_stats(result["compression"])
            _print_context(manager)
    elif command == "/rename":
        if not args:
            print("Usage: /rename <name>")
        else:
            manager.rename(" ".join(args))
            print("Session renamed.")
    elif command == "/delete":
        if not args:
            print("Usage: /delete <id|name>")
        else:
            manager.delete(" ".join(args))
            print("Session deleted.")
    elif command == "/plan":
        if manager.session is None:
            print("No active session.")
        else:
            print(json.dumps(manager.session.load_plan(), ensure_ascii=False, indent=2))
    elif command == "/memory":
        if manager.session is None:
            print("No active session.")
        else:
            session = manager.session
            print(f"memory: {session.memory_path}")
            print(f"plan:   {session.plan_path}")
            print(f"steps:  {session.run_path}")
            meta = session.load_memory_meta()
            if meta:
                print("meta:   " + json.dumps(meta, ensure_ascii=False))
            preview = session.load_memory().strip()
            print()
            print(preview[:4000] if preview else "(empty)")
            _print_context(manager)
    elif command == "/model":
        if not args or args[0] == "list":
            for item in manager.list_models():
                marker = "*" if item["current"] else " "
                label = item["label"]
                if item["chat_only"]:
                    label += " [chat-only]"
                style = "red" if item["chat_only"] else ""
                CONSOLE.print(f"{marker} {label}", style=style)
        else:
            label = " ".join(args)
            manager.set_model(label)
            print(f"Model set to {manager.model_label}.")
            _print_context(manager)
    elif command == "/compress-model":
        if not args or args[0] == "list":
            for item in manager.list_compression_models():
                marker = "*" if item["current"] else " "
                print(f"{marker} {item['label']}")
        else:
            label = manager.set_compression_model(" ".join(args))
            if label:
                print(f"Compression model set to {label}.")
            else:
                print("Compression model follows the session model.")
    elif command == "/tools":
        action = args[0] if args else "list"
        if action == "list":
            info = manager.list_tools()
            print("registered: " + ", ".join(info["registered"]))
            print("builtin:    " + ", ".join(info["builtin"]))
            print("active:     " + (", ".join(info["active"]) or "(none)"))
        elif action == "add":
            if len(args) != 2:
                print("Usage: /tools add <name|*>")
            elif args[1] == "*":
                result = manager.add_all_tools()
                added = ", ".join(result["added"]) or "(none)"
                skipped = ", ".join(result["skipped"]) or "(none)"
                print(f"Added: {added}")
                print(f"Skipped: {skipped}")
            else:
                manager.add_tool(args[1])
                print(f"Tool '{args[1]}' attached.")
        elif action == "remove":
            if len(args) != 2:
                print("Usage: /tools remove <name>")
            else:
                manager.remove_tool(args[1])
                print(f"Tool '{args[1]}' detached.")
        elif action == "clear":
            manager.clear_tools()
            print("All tools detached.")
        else:
            print("Usage: /tools [list|add|remove|clear] ...")
    elif command == "/experts":
        action = args[0] if args else "list"
        if action == "list":
            experts = manager.list_experts()
            if not experts:
                print("No experts configured.")
            for item in experts:
                state = "on" if item["active"] else "off"
                tools = ", ".join(item["tools"]) or "(none)"
                expertise = item["expertise"] or "(none)"
                print(
                    f"{state:3} {item['role']}  expertise={expertise}  "
                    f"model={item['model']}  "
                    f"allow_tools={item['allow_tools']}  tools={tools}"
                )
        elif action in {"enable", "disable"}:
            if len(args) != 2:
                print(f"Usage: /experts {action} <role|*>")
            elif args[1] == "*":
                if action == "enable":
                    roles = manager.enable_all_experts()
                    print("Experts enabled: " + (", ".join(roles) or "(none)"))
                else:
                    manager.disable_all_experts()
                    print("All experts disabled.")
            elif action == "enable":
                manager.enable_expert(args[1])
                print(f"Expert '{args[1]}' enabled.")
            else:
                manager.disable_expert(args[1])
                print(f"Expert '{args[1]}' disabled.")
        elif action == "tools":
            if len(args) < 2:
                print("Usage: /experts tools <role> [tool ...|none]")
            else:
                role = args[1]
                names = args[2:]
                if not names:
                    item = next(
                        (
                            expert
                            for expert in manager.list_experts()
                            if expert["role"] == role
                        ),
                        None,
                    )
                    if item is None:
                        print(f"Unknown expert: {role}")
                    else:
                        tools = ", ".join(item["tools"]) or "(none)"
                        print(
                            f"{role}: allow_tools={item['allow_tools']} "
                            f"tools={tools}"
                        )
                else:
                    if names == ["none"]:
                        names = []
                    manager.set_expert_tools(role, names)
                    print(f"Tools for expert '{role}' updated.")
        else:
            print("Usage: /experts [list|enable|disable|tools] ...")
    elif command == "/compress":
        summary = manager.compress()
        print(summary if summary else "(nothing to compress)")
        if manager.last_compression:
            _print_compression(manager)
            manager.last_compression = {}
    elif command == "/context":
        _print_context_details(manager)
    elif command == "/reset":
        manager.reset(hard="--hard" in args)
        print("Session history reset.")
        _print_context(manager)
    else:
        print(f"Unknown command: {command}. Type /help.")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avocado")
    parser.add_argument(
        "--config",
        help="Path to config.json (processed by the avocado entry point)",
    )
    parser.add_argument("--model", help="Model label, e.g. Groq/openai/gpt-oss-20b")
    parser.add_argument("--session", help="Session id or name to open")
    parser.add_argument("--sessions-dir", type=Path, help="Sessions storage root")
    parser.add_argument("--once", help="Run one task, then save and exit")
    parser.add_argument(
        "--no-compress-on-exit",
        action="store_true",
        help="Do not compress the step journal when exiting",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = SessionStore(root=args.sessions_dir)
    manager = SessionManager(
        store=store,
        model_label=args.model,
        compress_on_exit=not args.no_compress_on_exit,
    )

    try:
        session = manager.open(args.session)
        render_banner(
            session,
            manager.model_label,
            manager.compression_model_label,
        )
        _print_context(manager)
    except Exception as error:
        CONSOLE.print(f"Error: {error}", style="red")
        return 1

    if args.once:
        try:
            result = manager.run(args.once)
            render_expert_calls(manager.last_expert_calls)
            render_answer(result)
            _print_context(manager)
        except Exception as error:
            CONSOLE.print(f"Error: {error}", style="red")
            return 1
        finally:
            manager.close()
            if manager.last_warning:
                CONSOLE.print(f"Warning: {manager.last_warning}", style="yellow")
            _print_compression(manager)
        return 0

    render_status("Type /help for commands.")
    running = True
    try:
        while running:
            try:
                line = input("avocado> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                try:
                    running = _handle_command(manager, line)
                except Exception as error:
                    CONSOLE.print(f"Error: {error}", style="red")
                continue
            try:
                result = manager.run(line)
                render_expert_calls(manager.last_expert_calls)
                render_answer(result)
                _print_context(manager)
            except Exception as error:
                CONSOLE.print(f"Error: {error}", style="red")
    finally:
        manager.close()
        if manager.last_warning:
            CONSOLE.print(f"Warning: {manager.last_warning}", style="yellow")
        _print_compression(manager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
