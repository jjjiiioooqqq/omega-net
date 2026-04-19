"""Run demonstrator and emit machine-readable output.

Compatible with PyCharm when project root is opened directly, even without an
editable install, by adding the local `src/` directory to `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engineering_intelligence.__main__ import main


if __name__ == "__main__":
    main()
