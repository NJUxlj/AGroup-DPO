"""Generate terminal-style screenshots for README."""

import os
from pathlib import Path

from _render import render_terminal

BASE = Path(__file__).resolve().parent


def main() -> None:
    render_terminal(
        [ln.rstrip("\n") for ln in open(BASE / "dpo_data_output.txt").readlines()],
        BASE / "dpo_data_gen.png",
        "DPO Data Pipeline — server6",
    )
    render_terminal(
        [ln.rstrip("\n") for ln in open(BASE / "dpo_train_output.txt").readlines()],
        BASE / "dpo_train.png",
        "DPO Smoke Training — server6",
    )


if __name__ == "__main__":
    main()
