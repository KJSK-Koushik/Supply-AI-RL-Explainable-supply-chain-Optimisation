"""Small helpers for building the final-review deck on the faculty template.

The template draws a green gradient header band, a footer band and a title
placeholder on every slide. New slides are made from the template's "Title and
Content" layout and the template's own decorative shapes are copied onto them,
so the new deck looks exactly like the template the faculty supplied.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

GREEN = RGBColor(0x2E, 0x8B, 0x57)
GREEN_D = RGBColor(0x1B, 0x5E, 0x3A)
GREEN_L = RGBColor(0xE6, 0xF4, 0xEA)
PINK = RGBColor(0xD6, 0x33, 0x6C)
PINK_L = RGBColor(0xFD, 0xE8, 0xEF)
BLUE = RGBColor(0x2F, 0x6F, 0xB5)
BLUE_L = RGBColor(0xE7, 0xF0, 0xFA)
AMBER = RGBColor(0xB7, 0x79, 0x1F)
AMBER_L = RGBColor(0xFD, 0xF3, 0xDC)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x5B, 0x65, 0x73)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
GREY_L = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

# Content area inside the template's header and footer bands, in inches.
CL, CT, CW, CH = 0.45, 1.5, 12.43, 5.4

CHROME = ("Rectangle 16", "Google Shape;91;p1", "Straight Connector 15")
FOOTER_TEXT = "[Final Project Review]"


def fill_chrome(src_slide, dst_slide) -> None:
    """Copy the header band, footer band and rule from a template slide."""
    tree = dst_slide.shapes._spTree
    pos = 2  # after nvGrpSpPr and grpSpPr, i.e. behind everything else
    for shp in src_slide.shapes:
        if shp.name in CHROME:
            tree.insert(pos, copy.deepcopy(shp._element))
            pos += 1
        elif shp.name.startswith("Slide Number Placeholder"):
            tree.append(copy.deepcopy(shp._element))
    for shp in dst_slide.shapes:
        if shp.name == "Google Shape;91;p1":
            runs = [r for p in shp.text_frame.paragraphs for r in p.runs]
            runs[0].text = FOOTER_TEXT
            for r in runs[1:]:
                r.text = ""


def new_slide(prs, src_slide, title: str, size: int = 32):
    layout = next(layout for layout in prs.slide_layouts if layout.name == "Title and Content")
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx not in (0,):
            ph._element.getparent().remove(ph._element)
    fill_chrome(src_slide, s)
    tf = s.shapes.title.text_frame
    tf.text = title
    for r in tf.paragraphs[0].runs:
        r.font.size = Pt(size)
    return s


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def _style_run(run, size, bold=False, color=INK, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, margin=0.05):
    """paras: list of dicts or strings. dict keys: t, size, bold, color, bullet, space, italic,
    runs=[(text, {bold, color})] for mixed formatting."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for p in paras:
        if isinstance(p, str):
            p = {"t": p}
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get("align", align)
        para.space_after = Pt(p.get("space", 4))
        size = p.get("size", 16)
        bullet = p.get("bullet", False)
        lead = "•  " if bullet else ""
        runs = p.get("runs") or [(p.get("t", ""), {})]
        for i, (t, fmt) in enumerate(runs):
            r = para.add_run()
            r.text = (lead if i == 0 else "") + t
            _style_run(
                r,
                fmt.get("size", size),
                fmt.get("bold", p.get("bold", False)),
                fmt.get("color", p.get("color", INK)),
                fmt.get("italic", p.get("italic", False)),
            )
    return box


def card(slide, x, y, w, h, title, body, fill=GREY_L, accent=GREEN, title_size=17, body_size=14):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.adjustments[0] = 0.07
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = accent
    shape.line.width = Pt(1.75)
    shape.shadow.inherit = False
    shape.text_frame.text = ""
    paras = [{"t": title, "size": title_size, "bold": True, "color": accent, "space": 5}]
    for b in body:
        paras.append(
            b if isinstance(b, dict) else {"t": b, "size": body_size, "bullet": True, "space": 3}
        )
    text(slide, x + 0.12, y + 0.08, w - 0.24, h - 0.16, paras)
    return shape


def stat(slide, x, y, w, h, value, label, fill=GREEN, color=WHITE, value_size=30, label_size=13):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.adjustments[0] = 0.12
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    text(
        slide,
        x + 0.08,
        y + 0.05,
        w - 0.16,
        h - 0.1,
        [
            {
                "t": value,
                "size": value_size,
                "bold": True,
                "color": color,
                "align": PP_ALIGN.CENTER,
                "space": 1,
            },
            {"t": label, "size": label_size, "color": color, "align": PP_ALIGN.CENTER},
        ],
        anchor=MSO_ANCHOR.MIDDLE,
    )


def picture(slide, path, x, y, w, h, border=True, crop=None, caption=None, cap_size=12):
    """Fit an image inside the box (x, y, w, h) inches, centred, keeping aspect.
    crop = (left, top, right, bottom) pixels of the source image."""
    path = Path(path)
    im = Image.open(path)
    if crop:
        im = im.crop(crop)
        tmp = Path(tempfile.gettempdir()) / f"_crop_{path.stem}_{abs(hash(crop)) % 10**8}.png"
        im.save(tmp)
        path = tmp
    iw, ih = im.size
    cap_h = 0.32 if caption else 0.0
    scale = min(w / iw, (h - cap_h) / ih)
    pw, ph = iw * scale, ih * scale
    px, py = x + (w - pw) / 2, y
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    if border:
        pic.line.color.rgb = LINE
        pic.line.width = Pt(1)
    if caption:
        text(
            slide,
            x,
            py + ph + 0.03,
            w,
            cap_h,
            [{"t": caption, "size": cap_size, "color": MUTED, "align": PP_ALIGN.CENTER}],
        )
    return pic


def table(
    slide,
    data,
    x,
    y,
    w,
    col_w,
    font=13,
    head_font=None,
    row_h=0.36,
    bold_rows=(),
    hl_rows=(),
    align_right_from=1,
):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(
        rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows)
    )
    tbl = shape.table
    total = sum(col_w)
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    for r in range(rows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.text = ""
            para = tf.paragraphs[0]
            run = para.add_run()
            run.text = str(data[r][c])
            head = r == 0
            _style_run(
                run,
                (head_font or font) if head else font,
                bold=head or r in bold_rows,
                color=WHITE if head else INK,
            )
            if c >= align_right_from and not head:
                para.alignment = PP_ALIGN.CENTER
            if head:
                para.alignment = PP_ALIGN.CENTER if c >= align_right_from else PP_ALIGN.LEFT
            cell.fill.solid()
            if head:
                cell.fill.fore_color.rgb = GREEN
            elif r in hl_rows:
                cell.fill.fore_color.rgb = GREEN_L
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else GREY_L
    return shape


def badge(slide, x, y, n, color=GREEN, d=0.36):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.text = ""
    r = tf.paragraphs[0].add_run()
    r.text = str(n)
    _style_run(r, 13, bold=True, color=WHITE)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def trim_white(path: Path, out: Path, pad: int = 6) -> Path:
    """Crop uniform white space off an infographic's right and bottom edges."""
    from PIL import ImageChops

    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im = im.crop((0, 0, im.width, min(im.height, box[3] + pad)))
    im.save(out)
    return out
