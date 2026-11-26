"""Run one of the ExAbs evaluation environments."""

from __future__ import annotations

import argparse
import importlib
import sys

SCENES = {
    "cube": "cube.evaluate",
    "two-room": "two_room.evaluate",
    "counter-flow": "counter_flow.evaluate",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an ExAbs checkpoint on a packaged task suite.",
        usage="%(prog)s {cube,two-room,counter-flow} [scene options]",
    )
    parser.add_argument("scene", choices=SCENES)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"{sys.argv[0]} {args.scene}", *sys.argv[2:]]
    importlib.import_module(SCENES[args.scene]).main()


if __name__ == "__main__":
    main()
