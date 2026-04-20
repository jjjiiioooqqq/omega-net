"""CLI entrypoint for the engineering intelligence web dashboard."""

from __future__ import annotations

import argparse

from engineering_intelligence.web import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run engineering_intelligence web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
