import os
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--config":
            if index + 1 >= len(args):
                print("Error: --config requires a path.", file=sys.stderr)
                return 2
            os.environ["AVOCADO_CONFIG_PATH"] = args[index + 1]
            del args[index : index + 2]
            continue
        if arg.startswith("--config="):
            os.environ["AVOCADO_CONFIG_PATH"] = arg.split("=", 1)[1]
            del args[index]
            continue
        index += 1

    try:
        from .config import config_loader  # noqa: F401
    except Exception as error:
        if error.__class__.__name__ == "ConfigError":
            print(f"Error: {error}", file=sys.stderr)
            return 1
        raise

    from .cli import main as cli_main

    return cli_main(args)
