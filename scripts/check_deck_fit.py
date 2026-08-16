"""Estimate whether any text overflows its shape.

There is no LibreOffice in this environment, so slides cannot be rendered for
visual inspection. This approximates the one defect that renders worst -- text
spilling past its container -- from geometry alone: characters per line from the
box width, lines from the wrap, height from the line count.

Deliberately pessimistic (wide characters, generous indents, no hyphenation) so
that "fits" here means "fits in PowerPoint", not the other way round.
"""

from __future__ import annotations

import re
import zipfile

EMU_PT = 12700
CHAR_W = 0.53  # average glyph width as a fraction of point size
LINE_H = 1.22  # line height as a multiple of point size
INDENT = {0: 400000, 1: 850000}  # bullet indent per outline level
SLIDE_H = 6858000
SLIDE_W = 12192000
FOOTER_Y = 6416040  # template's bottom band -- content must stay above it

z = zipfile.ZipFile("SupplyAI-RL_Mid_Project_Review.pptx")
problems = []


def lines_for(text: str, size_pt: float, width_emu: int) -> int:
    per_line = max(1, int(width_emu / (size_pt * CHAR_W * EMU_PT)))
    # Wrap on whole words rather than dividing the character count, which
    # understates the line count whenever a long word is pushed down.
    n, cur = 1, 0
    for word in text.split():
        add = len(word) + (1 if cur else 0)
        if cur + add > per_line:
            n += 1
            cur = len(word)
        else:
            cur += add
    return n


def check_textboxes(slide: str, xml: str) -> None:
    # Strip tables and pictures first. They are not <p:sp>, so on a naive split
    # they ride inside the preceding shape's chunk and their geometry gets read
    # as if it were that shape's text box.
    xml = re.sub(r"<p:graphicFrame>.*?</p:graphicFrame>", "", xml, flags=re.S)
    xml = re.sub(r"<p:pic>.*?</p:pic>", "", xml, flags=re.S)
    for sp in xml.split("<p:sp>")[1:]:
        xf = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', sp)
        if not xf:
            continue
        x, y, cx, cy = (int(v) for v in xf.groups())
        if cy < 200000:
            continue
        # The template paints a full-width band behind every slide and a footer
        # strip along the bottom. Both legitimately span the slide, so measuring
        # them against the footer line reports the design as a defect.
        if cx >= SLIDE_W - 10000:
            continue
        paras = re.findall(r"<a:p>(.*?)</a:p>", sp, re.S)
        total = 0.0
        for p in paras:
            lvl = int(m.group(1)) if (m := re.search(r'lvl="(\d+)"', p)) else 0
            sz = int(m.group(1)) / 100 if (m := re.search(r'sz="(\d+)"', p)) else 18.0
            txt = " ".join(re.findall(r"<a:t>(.*?)</a:t>", p, re.S))
            if not txt.strip():
                continue
            # Diagram shapes suppress the bullet and set their own line
            # spacing. Charging them a bullet indent they do not have, and a
            # paragraph gap they did not ask for, made the checker report every
            # one of them as overflowing -- which is how a checker stops being
            # believed.
            bulleted = "<a:buNone/>" not in p
            avail = cx - (INDENT.get(lvl, 850000) if bulleted else 182880)
            spc = (
                int(m.group(1)) / 100000
                if (m := re.search(r'<a:lnSpc><a:spcPct val="(\d+)"/>', p))
                else 1.0
            )
            total += lines_for(txt, sz, avail) * sz * LINE_H * spc * EMU_PT
            if m := re.search(r'<a:spcBef><a:spcPts val="(\d+)"/>', p):
                total += int(m.group(1)) / 100 * EMU_PT
        if total > cy * 1.02:
            problems.append(
                f"{slide}: text box at y={y} needs ~{total / 914400:.2f}in "
                f"but is {cy / 914400:.2f}in tall"
            )
        if y + cy > FOOTER_Y:
            problems.append(f"{slide}: text box bottom {y + cy} crosses footer {FOOTER_Y}")
        if x + cx > SLIDE_W:
            problems.append(f"{slide}: text box right edge {x + cx} past slide {SLIDE_W}")


def check_tables(slide: str, xml: str) -> None:
    for gf in re.findall(r"<p:graphicFrame>(.*?)</p:graphicFrame>", xml, re.S):
        xf = re.search(r'<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', gf)
        if not xf:
            continue
        x, y, cx, cy = (int(v) for v in xf.groups())
        widths = [int(w) for w in re.findall(r'<a:gridCol w="(\d+)"', gf)]
        rows = re.findall(r"<a:tr h=\"(\d+)\">(.*?)</a:tr>", gf, re.S)
        used = 0
        for dec_h, row in rows:
            cells = re.findall(r"<a:tc>(.*?)</a:tc>", row, re.S)
            tallest = 0.0
            for i, c in enumerate(cells):
                if i >= len(widths):
                    continue
                sz = int(m.group(1)) / 100 if (m := re.search(r'sz="(\d+)"', c)) else 18.0
                txt = " ".join(re.findall(r"<a:t>(.*?)</a:t>", c, re.S))
                avail = widths[i] - 137160  # left + right cell margins
                h = lines_for(txt, sz, avail) * sz * LINE_H * EMU_PT + 68580
                tallest = max(tallest, h)
            used += max(float(dec_h), tallest)
        if x + sum(widths) > SLIDE_W:
            problems.append(f"{slide}: table right edge {x + sum(widths)} past slide")
        if y + used > FOOTER_Y:
            problems.append(
                f"{slide}: table needs {used / 914400:.2f}in from y={y / 914400:.2f}in, "
                f"bottom {(y + used) / 914400:.2f}in crosses footer at {FOOTER_Y / 914400:.2f}in"
            )


def check_pictures(slide: str, xml: str) -> None:
    for pic in re.findall(r"<p:pic>(.*?)</p:pic>", xml, re.S):
        xf = re.search(r'<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', pic)
        if not xf:
            continue
        x, y, cx, cy = (int(v) for v in xf.groups())
        if x + cx > SLIDE_W or y + cy > FOOTER_Y:
            problems.append(f"{slide}: picture extends past slide/footer")


names = sorted(
    (n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
    key=lambda n: int(re.findall(r"\d+", n)[0]),
)
for n in names:
    xml = z.read(n).decode("utf8")
    label = n.split("/")[-1].replace(".xml", "")
    check_textboxes(label, xml)
    check_tables(label, xml)
    check_pictures(label, xml)

if problems:
    print("POTENTIAL OVERFLOW:")
    for p in problems:
        print("  -", p)
else:
    print("no overflow detected")
