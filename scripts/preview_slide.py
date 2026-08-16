"""Draw a slide from its own XML so the layout can be looked at.

There is no LibreOffice here, so this is the only way to see whether the
diagram reads. It is an approximation -- PowerPoint's text metrics and
autofit are not reproduced -- so it is used to catch layout faults (collisions,
imbalance, a card that is clearly too small for its text), not to sign off on
exact glyph positions.
"""

from __future__ import annotations

import re
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

EMU_IN = 914400
SLIDE_W, SLIDE_H = 12192000, 6858000

deck, slide, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(deck)
xml = z.read(f"ppt/slides/{slide}.xml").decode("utf8")

fig, ax = plt.subplots(figsize=(SLIDE_W / EMU_IN, SLIDE_H / EMU_IN), dpi=110)
ax.set_xlim(0, SLIDE_W)
ax.set_ylim(SLIDE_H, 0)
ax.axis("off")
ax.add_patch(mpatches.Rectangle((0, 0), SLIDE_W, SLIDE_H, fc="white", ec="none"))


def runs(sp: str):
    for m in re.finditer(
        r'<a:rPr[^>]*?sz="(\d+)"[^>]*?b="([01])"[^>]*?>(?:(?!</a:r>).)*?'
        r'srgbClr val="([0-9A-Fa-f]{6})"(?:(?!</a:r>).)*?<a:t>(.*?)</a:t>',
        sp,
        re.S,
    ):
        yield int(m.group(1)) / 100, m.group(2) == "1", m.group(3), m.group(4)


for sp in xml.split("<p:sp>")[1:]:
    xf = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', sp)
    if not xf:
        continue
    x, y, w, h = (int(v) for v in xf.groups())
    if w >= SLIDE_W - 10000 and h < 1400000:
        continue
    fill = re.search(r'<a:solidFill><a:srgbClr val="([0-9A-Fa-f]{6})"/>', sp)
    edge = re.search(r'<a:ln w="\d+"><a:solidFill><a:srgbClr val="([0-9A-Fa-f]{6})"', sp)
    is_arrow = "Arrow" in sp[:200]
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=60000",
            fc="#" + (fill.group(1) if fill else "FFFFFF"),
            ec=("#" + edge.group(1)) if edge else ("none" if is_arrow else "#999999"),
            lw=1.0,
        )
    )
    rs = list(runs(sp))
    if not rs:
        continue
    total = sum(pt * 1.35 * 12700 for pt, _, _, _ in rs)
    anchor_top = 'anchor="t"' in sp
    cy = y + 60000 if anchor_top else y + (h - total) / 2
    for pt, bold, col, txt in rs:
        lh = pt * 1.35 * 12700
        algn_left = 'algn="l"' in sp
        ax.text(
            x + (90000 if algn_left else w / 2),
            cy + lh / 2,
            txt,
            ha="left" if algn_left else "center",
            va="center",
            fontsize=pt * 1.02,
            fontweight="bold" if bold else "normal",
            color="#" + col,
            wrap=True,
        )
        cy += lh

fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
