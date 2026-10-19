import argparse
import json
import shlex
from pathlib import Path
from typing import Optional, Sequence

from .sessions.manager import SessionManager
from .sessions.store import SessionStore

HELP_TEXT = """Commands:
  /help                 show this help
  /sessions             list saved sessions
  /new [name]           create and switch to a new session
  /switch <id|name>     compress the current session, then switch
  /rename <name>        rename the current session
  /delete <id|name>     delete a non-active session
  /plan                 show the current plan
  /memory               show memory paths and preview
  /tools [list]         list registered and active tools
  /tools add <name>     attach a registered tool to this session
  /tools remove <name>  detach a tool from this session
  /experts [list]       list experts and their tools
  /experts enable <role>    attach an expert to this session
  /experts disable <role>   detach an expert from this session
  /experts tools <role> [tool ...|none]
                        set the tools an expert may use
  /compress             fold the raw step journal into memory.md
  /reset [--hard]       clear current history; --hard also clears memory
  /quit                 save and exit

Any other input is sent to the main agent.
"""


def _print_session(manager: SessionManager) -> None:
    session = manager.session
    if session is None:
        return
    print(
        f"Session: {session.name} ({session.session_id}) "
        f"model={manager.model_label} path={session.path}"
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
    elif command == "/switch":
        if not args:
            print("Usage: /switch <id|name>")
        else:
            result = manager.switch(" ".join(args))
            session = result["session"]
            print(f"Switched to {session.name} ({session.session_id}).")
            if result["compressed"]:
                print("Previous session was compressed.")
            if result["warning"]:
                print(f"Warning: {result['warning']}")
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
    elif command == "/tools":
        action = args[0] if args else "list"
        if action == "list":
            info = manager.list_tools()
            print("registered: " + ", ".join(info["registered"]))
            print("active:     " + (", ".join(info["active"]) or "(none)"))
        elif action == "add":
            if len(args) != 2:
                print("Usage: /tools add <name>")
            else:
                manager.add_tool(args[1])
                print(f"Tool '{args[1]}' attached.")
        elif action == "remove":
            if len(args) != 2:
                print("Usage: /tools remove <name>")
            else:
                manager.remove_tool(args[1])
                print(f"Tool '{args[1]}' detached.")
        else:
            print("Usage: /tools [list|add|remove] ...")
    elif command == "/experts":
        action = args[0] if args else "list"
        if action == "list":
            experts = manager.list_experts()
            if not experts:
                print("No experts configured.")
            for item in experts:
                state = "on" if item["active"] else "off"
                tools = ", ".join(item["tools"]) or "(none)"
                print(
                    f"{state:3} {item['role']}  "
                    f"allow_tools={item['allow_tools']}  tools={tools}"
                )
        elif action in {"enable", "disable"}:
            if len(args) != 2:
                print(f"Usage: /experts {action} <role>")
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
    elif command == "/reset":
        manager.reset(hard="--hard" in args)
        print("Session history reset.")
    else:
        print(f"Unknown command: {command}. Type /help.")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avocado")
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
        manager.open(args.session)
        _print_session(manager)
    except Exception as error:
        print(f"Error: {error}")
        return 1

    if args.once:
        try:
            print(manager.run(args.once))
        except Exception as error:
            print(f"Error: {error}")
            return 1
        finally:
            manager.close()
            if manager.last_warning:
                print(f"Warning: {manager.last_warning}")
        return 0

    print("Type /help for commands.")
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
                    print(f"Error: {error}")
                continue
            try:
                result = manager.run(line)
                print(result)
            except Exception as error:
                print(f"Error: {error}")
    finally:
        manager.close()
        if manager.last_warning:
            print(f"Warning: {manager.last_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
