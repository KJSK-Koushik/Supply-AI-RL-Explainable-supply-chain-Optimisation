"""Capture the dashboard screenshots used as evidence in the final-review deck.

    python scripts/capture_dashboard.py

Drives headless Chrome over the DevTools protocol (the ``websockets`` package
is already in the venv). The dashboard must be running on localhost:8501.

Each shot is a full-page capture at a fixed tall viewport, so Streamlit's
sidebar and layout never reflow mid-capture, then trimmed of the empty
space below the footer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets
from PIL import Image, ImageChops

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "final_deck_assets" / "screens"
PORT = 9444
WIDTH, HEIGHT = 1600, 2600
URL = "http://localhost:8501/?embed=true"
BUSY = "!!document.querySelector('[data-testid=\"stStatusWidget\"]')"

# name, tab index, policy to pick (None = default), crisis toggle, press Explain
SHOTS = [
    ("episode_engine", 0, None, False, False),
    ("episode_challenger", 0, "challenger", False, False),
    ("episode_crisis", 0, None, True, False),
    ("before_after", 1, None, False, False),
    ("explain", 2, None, False, True),
    ("comparison", 3, None, False, False),
    ("robustness", 4, None, False, False),
]


class CDP:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def send(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.send(
            "Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True
        )
        return r.get("result", {}).get("value")

    async def click(self, pt):
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.send(
                "Input.dispatchMouseEvent",
                type=kind,
                x=pt["x"],
                y=pt["y"],
                button="left",
                clickCount=1,
            )
            await asyncio.sleep(0.08)

    async def settle(self, extra=1.5):
        await asyncio.sleep(2.0)
        for _ in range(360):
            if not await self.js(BUSY):
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(extra)


COMBO_RECT = """
(() => {
  const el = document.querySelector('[role="combobox"][aria-label="Policy"]')
    || [...document.querySelectorAll('[role="combobox"]')][0];
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return {x: r.x + r.width / 2, y: r.y + r.height / 2};
})()
"""

OPTION_RECT = """
((needle) => {
  const o = [...document.querySelectorAll('[role="option"]')].find(x => x.innerText.includes(needle));
  if (!o) return {missing: [...document.querySelectorAll('[role="option"]')].map(x => x.innerText)};
  const r = o.getBoundingClientRect();
  return {x: r.x + r.width / 2, y: r.y + r.height / 2};
})(%s)
"""

CLICK_CRISIS = """
(() => {
  const i = document.querySelector('input[aria-label="Inject a crisis"]');
  if (!i) return false;
  i.click();
  return true;
})()
"""

BAND_DRAWN = """
[...document.querySelectorAll('.js-plotly-plot text')].some(
  t => t.textContent.includes('disruption active'))
"""

CLICK_EXPLAIN = """
(() => {
  const b = [...document.querySelectorAll('button')].find(x => x.innerText.includes('Explain this decision'));
  if (b) b.click();
  return !!b;
})()
"""


def trim(path: Path) -> None:
    """Cut the empty page below the footer, keeping a small margin. Only the
    main area is examined: the sidebar runs the full height of the window."""
    im = Image.open(path).convert("RGB")
    main = im.crop((420, 0, im.width, im.height))
    bg = Image.new("RGB", main.size, main.getpixel((main.width - 5, main.height - 5)))
    box = ImageChops.difference(main, bg).getbbox()
    if box:
        im.crop((0, 0, im.width, min(im.height, box[3] + 40))).save(path)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        proc = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                f"--remote-debugging-port={PORT}",
                f"--user-data-dir={profile}",
                f"--window-size={WIDTH},{HEIGHT}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            page = None
            for _ in range(60):
                try:
                    targets = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/json"))
                    page = next(t for t in targets if t["type"] == "page")
                    break
                except Exception:
                    time.sleep(0.3)
            ws = await websockets.connect(page["webSocketDebuggerUrl"], max_size=128 * 1024 * 1024)
            cdp = CDP(ws)
            await cdp.send("Page.enable")
            await cdp.send(
                "Emulation.setDeviceMetricsOverride",
                width=WIDTH,
                height=HEIGHT,
                deviceScaleFactor=1,
                mobile=False,
            )
            wanted = sys.argv[1:]
            for name, tab, policy, crisis, explain in SHOTS:
                if wanted and name not in wanted:
                    continue
                # Fresh load every time: no state leaks between shots.
                await cdp.send("Page.navigate", url=URL)
                for _ in range(240):
                    await asyncio.sleep(0.5)
                    if await cdp.js(
                        "!!document.querySelector('[data-testid=\"stTabs\"]')"
                    ) and not await cdp.js(BUSY):
                        break
                await asyncio.sleep(1.0)
                if policy:
                    pt = None
                    for _ in range(40):  # the sidebar mounts a moment after the tabs
                        pt = await cdp.js(COMBO_RECT)
                        if pt:
                            break
                        await asyncio.sleep(0.5)
                    if not pt:
                        raise SystemExit("policy selector never appeared")
                    await cdp.click(pt)
                    await asyncio.sleep(1.0)
                    opt = await cdp.js(OPTION_RECT % json.dumps(policy))
                    print(name, "option:", opt)
                    await cdp.click(opt)
                    await cdp.settle()
                if crisis:
                    # The switch is a visually hidden checkbox: click the input itself.
                    for _ in range(40):
                        ok = await cdp.js(CLICK_CRISIS)
                        if ok:
                            break
                        await asyncio.sleep(0.5)
                    # The rerun simulates 180 days: wait until the disruption band
                    # is actually drawn (plotly text lives in SVG, not innerText).
                    seen = False
                    for _ in range(180):
                        await asyncio.sleep(1.0)
                        seen = await cdp.js(BAND_DRAWN)
                        if seen:
                            break
                    print(name, "disruption band drawn:", seen)
                    await asyncio.sleep(2.0)
                await cdp.js(f"document.querySelectorAll('[data-testid=\"stTab\"]')[{tab}].click()")
                await cdp.settle()
                if explain:
                    await cdp.js(CLICK_EXPLAIN)
                    for _ in range(300):
                        await asyncio.sleep(1.0)
                        if await cdp.js("!!document.querySelector('.sai-quote')"):
                            break
                    await asyncio.sleep(1.5)
                for _ in range(40):  # plotly draws after the status widget clears
                    drawn = await cdp.js(
                        "[...document.querySelectorAll('.js-plotly-plot')].filter("
                        "p => p.querySelector('.main-svg')).length"
                    )
                    total = await cdp.js("document.querySelectorAll('.js-plotly-plot').length")
                    # tab 2 (explain) has no charts; every other tab must show at least one
                    if drawn == total and (total > 0 or tab == 2):
                        break
                    await asyncio.sleep(0.5)
                await asyncio.sleep(2.0)
                shot = await cdp.send("Page.captureScreenshot", format="png", fromSurface=True)
                path = OUT / f"{name}.png"
                path.write_bytes(base64.b64decode(shot["data"]))
                trim(path)
                err = await cdp.js(
                    "[...document.querySelectorAll('[data-testid=\"stException\"]')]"
                    ".map(e => e.innerText).join(' | ')"
                )
                print("wrote", path.name, Image.open(path).size, "ERROR: " + err if err else "")
            await ws.close()
        finally:
            proc.kill()


asyncio.run(main())
