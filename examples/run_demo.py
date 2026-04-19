"""Run demonstrator and emit machine-readable output."""

from __future__ import annotations

import json
from pathlib import Path

from engineering_intelligence.demo.pipeline import run_demo


if __name__ == "__main__":
    result = run_demo()
    output_path = Path("examples") / "demo_audit_output.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
