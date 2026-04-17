from pathlib import Path

from engint.pipeline.demo import run_demo

if __name__ == "__main__":
    print(run_demo(Path("output_data/demo_audit.json")))
