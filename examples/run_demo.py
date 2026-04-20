from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engint.pipeline.demo import run_demo

if __name__ == "__main__":
    print(run_demo(Path("output_data/demo_audit.json")))
