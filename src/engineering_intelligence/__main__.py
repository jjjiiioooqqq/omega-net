"""CLI entrypoint for running the engineering intelligence demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engineering_intelligence.demo.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run engineering_intelligence demonstrator")
    parser.add_argument(
        "--output",
        default="examples/demo_audit_output.json",
        help="Path to write machine-readable demonstrator output JSON",
    )
    args = parser.parse_args()

    result = run_demo()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
