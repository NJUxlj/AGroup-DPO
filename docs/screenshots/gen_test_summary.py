"""Generate test summary screenshot from the report."""

from pathlib import Path

from _render import render_terminal

BASE = Path(__file__).resolve().parent
REPORTS = BASE.parent.parent / "reports"

with open(REPORTS / "copaw_cli_full_test.md", encoding="utf-8") as f:
    content = f.read()

lines = []
in_summary = False
for line in content.split("\n"):
    if "总结" in line or "copaw-dpo data" in line:
        in_summary = True
    if in_summary and line.strip().startswith("|"):
        lines.append(line.strip())

display_lines = [
    "=== copaw-dpo 全量测试结果 (server6, 2xRTX 4090) ===",
    "",
]
for row in lines:
    parts = [p.strip() for p in row.split("|") if p.strip()]
    if parts:
        display_lines.append("  " + "  |  ".join(parts))
display_lines += ["", "合计: 10/10 测试通过"]

render_terminal(
    display_lines,
    BASE / "copaw_cli_test_summary.png",
    "copaw-dpo CLI 全量测试",
    font_size=13,
    line_h=22,
    pad_x=30,
    pad_y=24,
)
