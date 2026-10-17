"""Rewrite the checked-in baseline and sample report from the fake model."""

from pathlib import Path

from llm_eval_obs.artifacts import BASELINE_PATH, HTML_PATH, MARKDOWN_PATH, write_demo_artifacts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    write_demo_artifacts(root)
    print(f"wrote {BASELINE_PATH}")
    print(f"wrote {MARKDOWN_PATH}")
    print(f"wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
