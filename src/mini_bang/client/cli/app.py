from __future__ import annotations

import sys

from mini_bang.client.cli.errors import CliError, EXIT_INTERNAL
from mini_bang.client.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return EXIT_INTERNAL
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"ERROR: internal failure: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
