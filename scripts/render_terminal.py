"""Turn captured console output into a terminal-style PNG.

    python scripts/render_terminal.py OUT.png "command as typed" captured_output.txt [captured_output2.txt ...]

Each text file is the real, unedited output of the command shown (captured with
``> file`` when the command was run). The script only draws it inside a dark
terminal window; it does not generate, alter or reorder any line. Several
(command, output) pairs can be stacked by repeating the arguments with ``--``:

    python scripts/render_terminal.py OUT.png "cmd one" out1.txt -- "cmd two" out2.txt
"""

from __future__ import annotations

import html
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ROOT = Path(__file__).resolve().parent.parent
PROMPT = f"PS {ROOT}> "

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1b1b1b; width: {width}px; font-family: Consolas, "Cascadia Mono", "Courier New", monospace; }}
  .win {{ margin: 18px; border-radius: 10px; overflow: hidden; background: #0c0c0c; border: 1px solid #3a3a3a; }}
  .bar {{ background: #202020; color: #d0d0d0; font: 13px "Segoe UI", Arial, sans-serif; padding: 9px 14px; display: flex; gap: 8px; align-items: center; }}
  .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
  .body {{ padding: 14px 18px 18px; color: #cccccc; font-size: {size}px; line-height: 1.32; white-space: pre; overflow: hidden; }}
  .p {{ color: #3b78ff; }} .c {{ color: #f2f2f2; }} .o {{ color: #cccccc; }}
</style></head><body><div class="win">
  <div class="bar"><span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span><span class="dot" style="background:#28c840"></span><span style="margin-left:10px">Windows PowerShell</span></div>
  <div class="body">{body}</div></div></body></html>"""


def block(cmd: str, text: str) -> str:
    out = html.escape(text.rstrip("\n"))
    return f'<span class="p">{html.escape(PROMPT)}</span><span class="c">{html.escape(cmd)}</span>\n<span class="o">{out}</span>\n\n'


def render(out: Path, pairs: list[tuple[str, str]], width: int = 1560, size: int = 15) -> None:
    body = "".join(block(c, t) for c, t in pairs).rstrip("\n") + "\n"
    lines = body.count("\n") + 1
    height = int(lines * size * 1.32) + 120
    page = PAGE.format(width=width, size=size, body=body)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = Path(tmp) / "t.html"
        src.write_text(page, encoding="utf-8")
        before = out.stat().st_mtime if out.exists() else 0.0
        proc = subprocess.Popen(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                f"--user-data-dir={tmp}/profile",
                "--force-device-scale-factor=2",
                f"--window-size={width},{height}",
                f"--screenshot={out}",
                src.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if out.exists() and out.stat().st_mtime > before:
                time.sleep(1.5)
                break
            time.sleep(0.5)
        else:
            proc.kill()
            raise SystemExit("no screenshot written")
        proc.kill()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        raise SystemExit(__doc__)
    target = Path(args[0]).resolve()
    groups: list[list[str]] = [[]]
    for a in args[1:]:
        if a == "--":
            groups.append([])
        else:
            groups[-1].append(a)
    items = []
    for g in groups:
        if len(g) < 2:
            raise SystemExit("each group needs a command and at least one output file")
        text = "\n".join(
            Path(f).read_text(encoding="utf-8", errors="replace").rstrip("\n") for f in g[1:]
        )
        items.append((g[0], text))
    render(target, items)
    print("wrote", target)
