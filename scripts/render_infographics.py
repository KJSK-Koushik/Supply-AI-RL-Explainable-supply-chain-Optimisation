"""Render the deck infographics (HTML in reports/final_deck_assets/src) to PNG.

    python scripts/render_infographics.py [name ...]

Uses headless Chrome at 2x so the images stay sharp on a projector. Each page
declares its own height on the .canvas element; the window is sized to match.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "final_deck_assets" / "src"
OUT = ROOT / "reports" / "final_deck_assets"
WIDTH = 1600


def canvas_height(html: str) -> int:
    m = re.search(r"\.canvas\s*\{[^}]*height:\s*(\d+)px", html)
    if not m:
        raise SystemExit("page must set .canvas { height: NNNpx }")
    return int(m.group(1))


def render(name: str) -> Path:
    page = SRC / f"{name}.html"
    height = canvas_height(page.read_text(encoding="utf-8"))
    out = OUT / f"{name}.png"
    before = out.stat().st_mtime if out.exists() else 0.0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        proc = subprocess.Popen(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                f"--user-data-dir={profile}",
                "--force-device-scale-factor=2",
                f"--window-size={WIDTH},{height}",
                f"--screenshot={out}",
                page.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Headless Chrome sometimes keeps running after it has written the
        # file. Wait for the file rather than for the process.
        deadline = time.time() + 90
        while time.time() < deadline:
            if out.exists() and out.stat().st_mtime > before and proc.poll() is not None:
                break
            if out.exists() and out.stat().st_mtime > before:
                time.sleep(1.5)  # let the write finish
                break
            time.sleep(0.5)
        else:
            proc.kill()
            raise SystemExit(f"no screenshot written for {name}")
        proc.kill()
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or [p.stem for p in sorted(SRC.glob("*.html"))]
    for n in names:
        print("wrote", render(n).relative_to(ROOT))
