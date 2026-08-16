"""Fill the 22AIE450 mid-review template with the SupplyAI-RL project content.

Works by string surgery on the unpacked template rather than python-pptx, which
is not installed here and in any case cannot preserve the template's styling
through a text assignment. Every slide keeps its own header band, footer,
title and slide-number placeholder exactly as the template author drew them;
only the empty content placeholder is filled.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

TPL = Path("tpl")
OUT = Path("SupplyAI-RL_Mid_Project_Review.pptx")

# Theme colours lifted from the template so added tables match the header band.
GREEN = "1A9B46"
DARKGREEN = "0D5A28"
INK = "202020"


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------- text bodies


def para(
    text: str,
    level: int = 0,
    size: int = 1800,
    bold: bool = False,
    color: str | None = None,
    bullet: bool = True,
) -> str:
    """One bulleted paragraph. Bullets are inherited from the layout unless
    explicitly suppressed -- a literal character here would render twice."""
    ppr = f'<a:pPr lvl="{level}">'
    ppr += '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
    ppr += '<a:spcBef><a:spcPts val="600"/></a:spcBef>'
    if not bullet:
        ppr += "<a:buNone/>"
    ppr += "</a:pPr>"
    rpr = f'<a:rPr lang="en-IN" sz="{size}" b="{1 if bold else 0}" dirty="0">'
    if color:
        rpr += f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
    rpr += "</a:rPr>"
    return f"<a:p>{ppr}<a:r>{rpr}<a:t>{esc(text)}</a:t></a:r></a:p>"


def body(items) -> str:
    """items: list of str, or (text, level), or (text, level, size, bold)."""
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(para(it))
        else:
            out.append(para(*it))
    return "".join(out)


# ------------------------------------------------------------ slide surgery


def split_shapes(xml: str) -> list[str]:
    return xml.split("<p:sp>")


def fill_content(xml: str, paragraphs: str, xfrm: tuple | None = None) -> str:
    """Replace the body of the shape holding <p:ph idx="1"/>."""
    parts = split_shapes(xml)
    for i, p in enumerate(parts):
        if '<p:ph idx="1"/>' not in p:
            continue
        new = re.sub(
            r"<p:txBody>.*?</p:txBody>",
            f"<p:txBody><a:bodyPr><a:normAutofit/></a:bodyPr><a:lstStyle/>{paragraphs}</p:txBody>",
            p,
            count=1,
            flags=re.S,
        )
        if xfrm:
            x, y, cx, cy = xfrm
            new = re.sub(
                r"<a:off x=\"\d+\" y=\"\d+\"/><a:ext cx=\"\d+\" cy=\"\d+\"/>",
                f'<a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/>',
                new,
                count=1,
            )
        parts[i] = new
        return "<p:sp>".join(parts)
    raise SystemExit("content placeholder not found")


def drop_graphicframe(xml: str) -> str:
    """Remove the template's placeholder table.

    Deliberately not done by splitting on <p:sp>: a graphicFrame is not a
    <p:sp>, so it rides along inside whichever shape's text precedes it, and
    dropping that chunk takes the slide title with it.
    """
    return re.sub(r"<p:graphicFrame>.*?</p:graphicFrame>", "", xml, count=1, flags=re.S)


# ------------------------------------------------------------------- tables


def cell(text: str, size: int, bold: bool, fill: str | None, color: str) -> str:
    tc = (
        "<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>"
        f'<a:p><a:pPr algn="l"/><a:r><a:rPr lang="en-IN" sz="{size}" '
        f'b="{1 if bold else 0}" dirty="0">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
        f"<a:t>{esc(text)}</a:t></a:r></a:p></a:txBody>"
    )
    if fill:
        tc += f'<a:tcPr marL="68580" marR="68580" marT="34290" marB="34290" anchor="ctr"><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill></a:tcPr>'
    else:
        tc += '<a:tcPr marL="68580" marR="68580" marT="34290" marB="34290" anchor="ctr"/>'
    return tc + "</a:tc>"


def table(
    shape_id: int,
    widths: list[int],
    rows: list[list[str]],
    x: int,
    y: int,
    size: int = 1100,
    row_h: int = 320000,
) -> str:
    """A graphicFrame table. Header row picks up the template's green."""
    grid = "".join(f'<a:gridCol w="{w}"/>' for w in widths)
    trs = []
    for r, row in enumerate(rows):
        head = r == 0
        cells = "".join(
            cell(
                c,
                size,
                head,
                GREEN if head else ("F2F7F3" if r % 2 == 0 else None),
                "FFFFFF" if head else INK,
            )
            for c in row
        )
        trs.append(f'<a:tr h="{row_h}">{cells}</a:tr>')
    cx = sum(widths)
    cy = row_h * len(rows)
    return (
        f"<p:graphicFrame><p:nvGraphicFramePr>"
        f'<p:cNvPr id="{shape_id}" name="Table {shape_id}"/>'
        f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr>'
        f"<p:nvPr/></p:nvGraphicFramePr>"
        f'<p:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></p:xfrm>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
        f'<a:tbl><a:tblPr firstRow="1" bandRow="1">'
        f"<a:tableStyleId>{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}</a:tableStyleId>"
        f"</a:tblPr><a:tblGrid>{grid}</a:tblGrid>{''.join(trs)}</a:tbl>"
        f"</a:graphicData></a:graphic></p:graphicFrame>"
    )


def add_shapes(xml: str, extra: str) -> str:
    return xml.replace("</p:spTree>", extra + "</p:spTree>")


def picture(shape_id: int, rid: str, x: int, y: int, cx: int, cy: int) -> str:
    return (
        f"<p:pic><p:nvPicPr>"
        f'<p:cNvPr id="{shape_id}" name="Picture {shape_id}"/>'
        f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/>'
        f"</p:nvPicPr>"
        f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="C8D8CC"/></a:solidFill></a:ln>'
        f"</p:spPr></p:pic>"
    )


def box(
    sid: int,
    x: int,
    y: int,
    cx: int,
    cy: int,
    lines: list[tuple],
    fill: str,
    line_col: str,
    dash: bool = False,
) -> str:
    """Rounded rectangle with vertically centred text. lines: (text, pt, bold, colour)."""
    paras = "".join(
        f'<a:p><a:pPr algn="ctr"><a:lnSpc><a:spcPct val="95000"/></a:lnSpc>'
        f'<a:buNone/></a:pPr><a:r><a:rPr lang="en-IN" sz="{int(pt * 100)}" '
        f'b="{1 if bold else 0}" dirty="0"><a:solidFill><a:srgbClr val="{col}"/>'
        f'</a:solidFill><a:latin typeface="Calibri"/></a:rPr>'
        f"<a:t>{esc(t)}</a:t></a:r></a:p>"
        for t, pt, bold, col in lines
    )
    # A txBody with no paragraph at all is rejected; shapes used purely as
    # rules or backing panels still need an empty one.
    paras = paras or "<a:p/>"
    ln = f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line_col}"/></a:solidFill>'
    ln += '<a:prstDash val="dash"/></a:ln>' if dash else "</a:ln>"
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Box {sid}"/>'
        f"<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 10000"/>'
        f"</a:avLst></a:prstGeom>"
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{ln}</p:spPr>'
        f'<p:txBody><a:bodyPr anchor="ctr" lIns="91440" rIns="91440" tIns="45720" '
        f'bIns="45720"><a:normAutofit/></a:bodyPr><a:lstStyle/>{paras}</p:txBody></p:sp>'
    )


def arrow(sid: int, x: int, y: int, cx: int, cy: int, down: bool = True) -> str:
    prst = "downArrow" if down else "rightArrow"
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Arrow {sid}"/>'
        f"<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="9BB7A5"/></a:solidFill>'
        f"<a:ln><a:noFill/></a:ln></p:spPr>"
        f"<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>"
    )


def textbox(sid: int, x: int, y: int, cx: int, cy: int, paras: str, align: str = "l") -> str:
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/>'
        f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0"><a:normAutofit/>'
        f"</a:bodyPr><a:lstStyle/>{paras}</p:txBody></p:sp>"
    )


def clone_slide(src: int, new: int, after: int, title: str) -> str:
    """Register a new slide cloned from `src`, ordered right after `after`.

    Slide order lives in <p:sldIdLst>, not in the file names, so the clone can
    keep any number as long as every part that refers to it agrees. Notes are
    deliberately not carried over -- the source's notesSlide belongs to the
    source, and pointing two slides at one notes part makes edits to either
    show up on both.
    """
    xml = read(src)
    xml = re.sub(
        r'(<p:ph type="title"/>.*?<a:t>).*?(</a:t>)',
        lambda m: m.group(1) + esc(title) + m.group(2),
        xml,
        count=1,
        flags=re.S,
    )
    # Drop everything the clone should not inherit: the source's body content.
    xml = drop_graphicframe(xml)
    write(new, xml)

    rels = TPL / f"ppt/slides/_rels/slide{new}.xml.rels"
    src_rels = (TPL / f"ppt/slides/_rels/slide{src}.xml.rels").read_text(encoding="utf8")
    src_rels = re.sub(r"<Relationship [^>]*notesSlide[^>]*/>", "", src_rels)
    rels.write_text(src_rels, encoding="utf8")

    ct = TPL / "[Content_Types].xml"
    c = ct.read_text(encoding="utf8")
    c = c.replace(
        "</Types>",
        f'<Override PartName="/ppt/slides/slide{new}.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        "</Types>",
    )
    ct.write_text(c, encoding="utf8")

    prels = TPL / "ppt/_rels/presentation.xml.rels"
    pr = prels.read_text(encoding="utf8")
    used = [int(i) for i in re.findall(r'Id="rId(\d+)"', pr)]
    rid = f"rId{max(used) + 1}"
    pr = pr.replace(
        "</Relationships>",
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/slide" Target="slides/slide{new}.xml"/>'
        "</Relationships>",
    )
    prels.write_text(pr, encoding="utf8")

    pres = TPL / "ppt/presentation.xml"
    p = pres.read_text(encoding="utf8")
    ids = [int(i) for i in re.findall(r'<p:sldId id="(\d+)"', p)]
    new_id = max(ids) + 1
    after_rid = re.search(
        rf'<Relationship Id="(rId\d+)"[^>]*Target="slides/slide{after}\.xml"', pr
    ).group(1)
    m = re.search(rf'<p:sldId id="(\d+)" r:id="{after_rid}"/>', p)
    p = p.replace(
        m.group(0),
        m.group(0) + f'<p:sldId id="{new_id}" r:id="{rid}"/>',
        1,
    )
    pres.write_text(p, encoding="utf8")
    return rid


def add_image_rel(slide_n: int, rid: str, filename: str, source: str) -> None:
    media = TPL / "ppt/media"
    media.mkdir(exist_ok=True)
    shutil.copy(source, media / filename)
    rels = TPL / f"ppt/slides/_rels/slide{slide_n}.xml.rels"
    r = rels.read_text(encoding="utf8")
    r = r.replace(
        "</Relationships>",
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/image" Target="../media/{filename}"/>'
        "</Relationships>",
    )
    rels.write_text(r, encoding="utf8")


def png_size(path) -> tuple[int, int]:
    import struct

    with open(path, "rb") as fh:
        head = fh.read(33)
    return struct.unpack(">II", head[16:24])


def read(n: int) -> str:
    return (TPL / f"ppt/slides/slide{n}.xml").read_text(encoding="utf8")


def write(n: int, xml: str) -> None:
    (TPL / f"ppt/slides/slide{n}.xml").write_text(xml, encoding="utf8")


CX = 10515600  # template content width
X = 838198  # template content left edge
Y = 1594415  # template content top edge

# ============================================================== slide 1 title

s = read(1)
# The title box is 12in x 1.14in and the template sets 40pt. This title is 49
# characters, which wraps to two lines at that size and overruns the box. The
# box carries normAutofit so PowerPoint would shrink it anyway, but relying on
# that means the size differs between PowerPoint and every other renderer --
# set it explicitly at 32pt, which fits on one line with room to spare.
s = s.replace(
    "<a:t>Project Title</a:t>",
    "<a:t>SupplyAI-RL: Explainable Supply Chain Optimization</a:t>",
)


def _retitle(m: re.Match) -> str:
    run = m.group(0)
    if re.search(r'\bsz="\d+"', run):
        # Overwrite the existing size. Inserting a second sz= would produce a
        # duplicate attribute, which is not well-formed XML and makes the file
        # unopenable -- PowerPoint reports it as corrupt rather than ignoring it.
        return re.sub(r'\bsz="\d+"', 'sz="3200"', run, count=1)
    return run.replace("<a:rPr ", '<a:rPr sz="3200" ', 1)


s = re.sub(
    r"<a:r>(?:(?!</a:r>).)*?<a:t>SupplyAI-RL(?:(?!</a:r>).)*?</a:r>",
    _retitle,
    s,
    count=1,
    flags=re.S,
)
s = s.replace(
    "<a:t>22AIE450 Speech Processing</a:t>",
    "<a:t>22AIE450 Reinforcement Learning</a:t>",
)
# Left as a bracketed prompt rather than invented names -- the team roster,
# batch number and presentation date are not derivable from the project.
s = s.replace(
    "<a:t>List of Team Members</a:t>",
    "<a:t>[Team Member Names and Roll Numbers]</a:t>",
)
write(1, s)

# ============================================================= slide 2 outline

write(
    2,
    fill_content(
        read(2),
        body(
            [
                "Introduction and Motivation",
                "Problem Definition",
                "Literature Review and Gaps Identified",
                "Objectives",
                "Methodology: simulator, RL formulation, LLM explanation layer",
                "Dataset Description and Calibration",
                "Preliminary Results: tuned classical policies vs the RL agent",
                "Interim Summary and Completion Plan",
            ]
        ),
    ),
)

# ======================================================== slide 3 motivation

write(
    3,
    fill_content(
        read(3),
        body(
            [
                (
                    "Retail inventory is a daily decision under uncertainty, and both errors are "
                    "expensive.",
                    0,
                    1800,
                    True,
                ),
                (
                    "Order too much: capital is tied up, storage is paid for, and unsold stock is "
                    "marked down or written off.",
                    1,
                    1500,
                ),
                (
                    "Order too little: the sale is lost permanently, because the customer buys "
                    "elsewhere.",
                    1,
                    1500,
                ),
                (
                    "Classical policies treat each product in isolation and assume demand is "
                    "well behaved.",
                    0,
                    1800,
                    True,
                ),
                (
                    "Real demand is not. In our data the variance of daily demand is roughly 216 "
                    "times its mean, so the standard Poisson assumption fails badly.",
                    1,
                    1500,
                ),
                (
                    "Reinforcement learning can express what a formula cannot: a joint ordering "
                    "and sourcing policy that adapts to season, volatility and supplier state.",
                    0,
                    1800,
                    True,
                ),
                ("But a manager will not act on a number they cannot interpret.", 0, 1800, True),
                (
                    "Hence the motivation: pair an RL decision-maker with an LLM that states the "
                    "reason for every order in business language.",
                    1,
                    1500,
                ),
            ]
        ),
    ),
)

# ==================================================== slide 4 problem definition

write(
    4,
    fill_content(
        read(4),
        body(
            [
                (
                    "Manage 10 products across 3 suppliers over a 180-day horizon, sharing a "
                    "single 20,000-unit warehouse.",
                    0,
                    1700,
                ),
                (
                    "Each simulated day, for every product, decide two things jointly: how much "
                    "to order, and from which supplier.",
                    0,
                    1700,
                ),
                (
                    "Suppliers differ in unit cost, lead time (1-8 days), fill reliability "
                    "(80-98%), minimum order quantity and outage risk.",
                    0,
                    1700,
                ),
                (
                    "Objective: maximise total profit = revenue - purchase - holding - ordering "
                    "fees - lost-sale penalty.",
                    0,
                    1700,
                    True,
                ),
                (
                    "Scale: 21 choices per product over 10 products is about 1.7 x 10^13 "
                    "combinations per day, so enumeration is impossible.",
                    0,
                    1700,
                ),
                (
                    "Decisions are coupled in time (today's order becomes next week's stock) and "
                    "across products (one shared capacity).",
                    0,
                    1700,
                ),
                (
                    "Additional requirement: every decision must come with a business-readable "
                    "justification.",
                    0,
                    1700,
                    True,
                ),
            ]
        ),
        xfrm=(X, Y, CX, 4351338),
    ),
)

# ==================================================== slide 5 literature review

s = drop_graphicframe(read(5))
lit = [
    [
        "Sl No",
        "Title and year",
        "Journal / Conference",
        "Methodology",
        "Key findings",
        "Limitations",
    ],
    [
        "1",
        "Proximal Policy Optimization Algorithms (2017)",
        "arXiv preprint",
        "Clipped surrogate policy-gradient objective",
        "Stable, sample-efficient on-policy RL; the default modern baseline",
        "General purpose; no supply-chain structure",
    ],
    [
        "2",
        "A Deep Q-Network for the Beer Game (2022)",
        "M&SOM",
        "DQN on a serial beer-game supply chain",
        "DRL matches or beats base-stock policies under cooperation",
        "Single product, serial chain, no sourcing choice",
    ],
    [
        "3",
        "Can Deep RL Improve Inventory Management? (2022)",
        "M&SOM",
        "A3C on lost-sales and dual-sourcing problems",
        "Competitive with tuned heuristics but not clearly superior",
        "Heavy tuning cost; no explanation layer",
    ],
    [
        "4",
        "Deep RL for Inventory Control: A Roadmap (2022)",
        "EJOR",
        "Survey and research roadmap",
        "Names fair benchmarking and interpretability as open problems",
        "Conceptual; no implementation or results",
    ],
    [
        "5",
        "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms (2022)",
        "FLAIRS",
        "Action masking inside policy-gradient methods",
        "Masking is a valid gradient and scales to large action spaces",
        "Not demonstrated on inventory or sourcing",
    ],
]
s = add_shapes(
    s,
    table(
        40,
        [520000, 2350000, 1500000, 2100000, 2300000, 1900000],
        lit,
        X,
        1700000,
        size=950,
        row_h=560000,
    ),
)
write(5, s)

# ========================================================== slide 6 gaps

write(
    6,
    fill_content(
        read(6),
        body(
            [
                (
                    "Weak benchmarks. Most DRL inventory papers compare against textbook-default "
                    "heuristics rather than properly tuned ones, which inflates the reported gain.",
                    0,
                    1700,
                ),
                (
                    "Narrow problem settings. Single-product or serial chains dominate; joint "
                    "quantity and supplier selection under one shared capacity is rarely modelled.",
                    0,
                    1700,
                ),
                (
                    "Explainability is deferred. Learned policies output a number, not a reason, "
                    "so they stay in simulation and never reach a decision-maker.",
                    0,
                    1700,
                ),
                (
                    "Stress testing is hand-built. Demand spikes and supplier failures are coded "
                    "case by case, so coverage is narrow and hard to extend.",
                    0,
                    1700,
                ),
                (
                    "No existing work couples an RL inventory agent with an LLM that both explains "
                    "its decisions and generates the scenarios used to test it.",
                    0,
                    1700,
                    True,
                ),
            ]
        ),
        xfrm=(X, Y, CX, 4351338),
    ),
)

# ======================================================== slide 7 objectives

write(
    7,
    fill_content(
        read(7),
        body(
            [
                (
                    "1. Build a supply chain simulator calibrated from real retail transaction "
                    "data.",
                    0,
                    1800,
                    True,
                ),
                (
                    "2. Train a reinforcement learning agent to choose reorder quantity and "
                    "supplier jointly.",
                    0,
                    1800,
                    True,
                ),
                ("3. Use an LLM to explain each decision in business language.", 0, 1800, True),
                (
                    "4. Use an LLM to generate demand-spike and supplier-delay scenarios for "
                    "stress testing.",
                    0,
                    1800,
                    True,
                ),
                (
                    "5. Compare the RL agent against properly tuned classical inventory policies.",
                    0,
                    1800,
                    True,
                ),
                (
                    "Objectives 1, 2 and 5 are complete. Objectives 3 and 4 are the next phase.",
                    0,
                    1500,
                ),
            ]
        ),
    ),
)

# ======================================================= slide 8 methodology

write(
    8,
    fill_content(
        read(8),
        body(
            [
                (
                    "Pipeline: real transactions - statistical calibration - simulator - agent "
                    "and baselines - held-out comparison - LLM explanation layer.",
                    0,
                    1600,
                ),
                ("MDP formulation", 0, 1700, True),
                (
                    "State (88 values): per product stock, in-transit, days of cover, 7-day mean "
                    "and volatility, trend, backlog; plus weekday, season, and each supplier's "
                    "availability, lead time and recent fill rate.",
                    1,
                    1350,
                ),
                (
                    "Action: for each product one joint choice of (quantity bucket, supplier) - "
                    "7 x 3 = 21 options, as MultiDiscrete over 10 products.",
                    1,
                    1350,
                ),
                ("Reward: that day's realised profit. Episode: 180 days.", 1, 1350),
                (
                    "Algorithm: MaskablePPO (PPO with invalid-action masking), MLP of two 128-unit "
                    "hidden layers, trained on CPU.",
                    0,
                    1700,
                    True,
                ),
                (
                    "Masking blocks suppliers in outage and orders exceeding free capacity. Worth "
                    "about 3x on its own: best profit rose from 23,717 to 72,740.",
                    1,
                    1350,
                ),
                (
                    "Fair comparison by construction: baselines are grid-search tuned, face the "
                    "same seeds and the same action granularity, and are scored on 30 held-out "
                    "seeds disjoint from all tuning and training seeds.",
                    0,
                    1700,
                    True,
                ),
            ]
        ),
        xfrm=(X, Y, CX, 4500000),
    ),
)

# ======================================================== slide 9 dataset

s = fill_content(
    read(9),
    body(
        [
            (
                "UCI Online Retail II: about 1 million transaction lines from a UK gift "
                "retailer, Dec 2009 - Dec 2011.",
                0,
                1450,
            ),
            (
                "Cleaning: cancellations, returns, negative quantities and missing product "
                "codes removed. Saturdays dropped entirely, as the retailer never traded "
                "then; zero-filling would teach a false weekly collapse.",
                0,
                1450,
            ),
            (
                "10 SKUs selected to span three demand regimes, so that no single fixed rule "
                "can serve them all.",
                0,
                1450,
            ),
            ("steady (CV 1.02-1.20), moderate (1.29-1.65), bursty (2.09-2.34)", 1, 1300),
            ("Calibrated only on data before 1 June 2011; later data is held back.", 0, 1450),
            (
                "Demand model: Gamma-Poisson (negative binomial), because variance is about "
                "216x the mean.",
                0,
                1450,
            ),
            (
                "A second dataset (Kaggle retail inventory) was screened and rejected: it "
                "showed no weekday effect and no seasonality, i.e. synthetic noise.",
                0,
                1450,
                True,
            ),
        ]
    ),
    xfrm=(X, Y, 5750000, 4351338),
)
s = add_shapes(s, picture(50, "rIdFig", 6750000, 1900000, 4600000, 3019000))
write(9, s)

rels9 = TPL / "ppt/slides/_rels/slide9.xml.rels"
r = rels9.read_text(encoding="utf8")
r = r.replace(
    "</Relationships>",
    '<Relationship Id="rIdFig" Type="http://schemas.openxmlformats.org/'
    'officeDocument/2006/relationships/image" Target="../media/figure1.png"/>'
    "</Relationships>",
)
rels9.write_text(r, encoding="utf8")

media = TPL / "ppt/media"
media.mkdir(exist_ok=True)
shutil.copy(
    Path(
        r"D:\SEM-7\Reinforcement leaning\SupplyAI-RL Explainable Supply Chain "
        r"Optimization\results\figures\04_demand_regimes.png"
    ),
    media / "figure1.png",
)

ct = TPL / "[Content_Types].xml"
c = ct.read_text(encoding="utf8")
if 'Extension="png"' not in c:
    c = c.replace("<Types ", "<Types ", 1)
    c = re.sub(
        r"(<Types[^>]*>)",
        r'\1<Default Extension="png" ContentType="image/png"/>',
        c,
        count=1,
    )
    ct.write_text(c, encoding="utf8")

# ============================================== slide 10 results: baselines

s = fill_content(
    read(10),
    body(
        [
            (
                "All five classical policies were tuned by grid search on 12 separate seeds, "
                "then scored on 30 held-out seeds. Profit is per 180-day episode, in GBP.",
                0,
                1400,
                False,
                None,
                False,
            ),
        ]
    ),
    xfrm=(X, Y, CX, 620000),
)
rows = [
    ["Policy", "Profit", "Std dev", "Fill rate", "Stockout cost", "Ordering cost"],
    ["forecast + safety stock", "83,624", "4,389", "97.9%", "5,821", "2,729"],
    ["newsvendor", "82,049", "4,222", "98.0%", "7,057", "3,275"],
    ["EOQ + reorder point", "72,302", "7,028", "96.8%", "9,988", "2,626"],
    ["(s, S) policy", "70,365", "6,782", "94.9%", "15,677", "2,905"],
    ["constant order (control)", "50,751", "12,723", "92.3%", "27,644", "3,334"],
    ["random (sanity check)", "-34,144", "12,971", "94.2%", "19,900", "14,869"],
]
s = add_shapes(
    s,
    table(
        41,
        [2900000, 1500000, 1350000, 1400000, 1750000, 1600000],
        rows,
        X,
        2300000,
        size=1200,
        row_h=430000,
    ),
)
s = add_shapes(
    s,
    (
        '<p:sp><p:nvSpPr><p:cNvPr id="42" name="Note"/><p:cNvSpPr txBox="1"/>'
        "<p:nvPr/></p:nvSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{X}" y="5450000"/>'
        '<a:ext cx="10515600" cy="700000"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0"><a:normAutofit/></a:bodyPr>'
        "<a:lstStyle/>"
        + para(
            "Tuning mattered: widening the search grid lifted the best policy from "
            "66,395 to 84,920 during tuning. Beating an untuned baseline would have "
            "proved nothing.",
            0,
            1300,
            False,
            DARKGREEN,
            False,
        )
        + "</p:txBody></p:sp>"
    ),
)
write(10, s)

# ========================================== slide 11 results: RL vs baseline

s = fill_content(
    read(11),
    body(
        [
            (
                "Same 30 held-out seeds, same action granularity, identical customers per "
                "seed, so the difference is skill and not luck.",
                0,
                1400,
                False,
                None,
                False,
            ),
        ]
    ),
    xfrm=(X, Y, CX, 620000),
)
rows = [
    ["Policy", "Profit", "Fill rate", "Ordering cost"],
    ["forecast + safety stock (best classical)", "83,624", "97.9%", "2,729"],
    ["MaskablePPO agent, 1M steps", "71,227", "94.3%", "7,068"],
    ["PPO without action masking, 1M steps", "22,566", "94.8%", "13,988"],
]
s = add_shapes(
    s, table(43, [4600000, 1900000, 1900000, 2100000], rows, X, 2250000, size=1250, row_h=450000)
)
s = add_shapes(
    s,
    (
        '<p:sp><p:nvSpPr><p:cNvPr id="44" name="Findings"/><p:cNvSpPr txBox="1"/>'
        "<p:nvPr/></p:nvSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{X}" y="4200000"/>'
        '<a:ext cx="10515600" cy="1900000"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0"><a:normAutofit/></a:bodyPr>'
        "<a:lstStyle/>"
        + para(
            "The classical policy still wins by 12,397 per episode "
            "(paired t = -9.22, n = 30, significant).",
            0,
            1500,
            True,
            DARKGREEN,
            False,
        )
        + para(
            "Where the gap comes from: stockouts about 4,700; fragmented sourcing "
            "about 4,300 (the agent pays 2.6x the baseline's ordering fees); lost "
            "revenue about 2,900.",
            0,
            1350,
            False,
            INK,
            False,
        )
        + para(
            "Action masking is the single largest lever found so far, tripling agent "
            "profit from 23,717 to 72,740.",
            0,
            1350,
            False,
            INK,
            False,
        )
        + para(
            "A hyperparameter sweep of 8 configurations over 6 million steps is "
            "running to test whether the remaining gap can be closed.",
            0,
            1350,
            False,
            INK,
            False,
        )
        + "</p:txBody></p:sp>"
    ),
)
write(11, s)

# ======================================================== slide 12 summary

write(
    12,
    fill_content(
        read(12),
        body(
            [
                (
                    "Built and verified: the simulator, four tuned classical policies, the RL "
                    "training pipeline, and a held-out evaluation harness. 92 automated tests "
                    "pass.",
                    0,
                    1600,
                ),
                (
                    "The dataset decision is evidence-based: one candidate dataset was screened "
                    "and rejected for having no weekday or seasonal structure.",
                    0,
                    1600,
                ),
                (
                    "Action masking is the largest single contributor to agent performance so far "
                    "(3x improvement).",
                    0,
                    1600,
                ),
                (
                    "Honest current finding: the tuned classical policy still beats the RL agent "
                    "by 12,397 per episode, and the difference is statistically significant.",
                    0,
                    1600,
                    True,
                ),
                (
                    "This is a meaningful result rather than an artefact, because the comparison "
                    "is fair by construction: same environment, same seeds, same action set.",
                    0,
                    1600,
                ),
                (
                    "Remaining work is the LLM explanation and scenario layer, which is the novel "
                    "contribution of the project.",
                    0,
                    1600,
                ),
            ]
        ),
        xfrm=(X, Y, CX, 4500000),
    ),
)

# =================================================== slide 13 completion plan

s = fill_content(
    read(13),
    body(
        [
            (
                "Engineering is complete; the remaining phases deliver the explainability "
                "contribution and the report.",
                0,
                1400,
                False,
                None,
                False,
            ),
        ]
    ),
    xfrm=(X, Y, CX, 560000),
)
rows = [
    ["Phase", "Work item", "Deliverable", "Status"],
    [
        "5",
        "Finalise held-out evaluation and hyperparameter sweep",
        "Final RL vs classical comparison table",
        "In progress",
    ],
    [
        "6",
        "LLM explanation module via OpenRouter",
        "Plain-English reason for every order decision",
        "Next",
    ],
    ["7", "LLM scenario generator", "Demand-spike and supplier-failure stress tests", "Planned"],
    ["8", "Streamlit decision-support dashboard", "Interactive interface for a manager", "Planned"],
    ["9", "LLM-guided curriculum training", "Agent trained on generated hard scenarios", "Planned"],
    [
        "10",
        "Final report and documentation",
        "Written report, figures, reproducible code",
        "Planned",
    ],
]
s = add_shapes(
    s, table(45, [900000, 3600000, 3900000, 1800000], rows, X, 2200000, size=1150, row_h=520000)
)
write(13, s)

# ======================================================== slide 14 references

write(
    14,
    fill_content(
        read(14),
        body(
            [
                (
                    "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). "
                    "Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
                    0,
                    1250,
                ),
                (
                    "Oroojlooyjadid, A., Nazari, M., Snyder, L. V., Takac, M. (2022). A Deep "
                    "Q-Network for the Beer Game. Manufacturing & Service Operations Management.",
                    0,
                    1250,
                ),
                (
                    "Gijsbrechts, J., Boute, R. N., Van Mieghem, J. A., Zhang, D. (2022). Can Deep "
                    "Reinforcement Learning Improve Inventory Management? Manufacturing & Service "
                    "Operations Management.",
                    0,
                    1250,
                ),
                (
                    "Boute, R. N., Gijsbrechts, J., van Jaarsveld, W., Vanvuchelen, N. (2022). "
                    "Deep Reinforcement Learning for Inventory Control: A Roadmap. European "
                    "Journal of Operational Research.",
                    0,
                    1250,
                ),
                (
                    "Huang, S., Ontanon, S. (2022). A Closer Look at Invalid Action Masking in "
                    "Policy Gradient Algorithms. FLAIRS-35.",
                    0,
                    1250,
                ),
                (
                    "Chen, D., Sain, S. L., Guo, K. (2012). Data mining for the online retail "
                    "industry. Journal of Database Marketing & Customer Strategy Management.",
                    0,
                    1250,
                ),
                (
                    "Dua, D., Graff, C. Online Retail II Data Set. UCI Machine Learning "
                    "Repository. https://archive.ics.uci.edu/dataset/502",
                    0,
                    1250,
                ),
                (
                    "Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning "
                    "Implementations. Journal of Machine Learning Research 22(268).",
                    0,
                    1250,
                ),
            ]
        ),
        xfrm=(X, Y, CX, 4600000),
    ),
)

# ================================================ slide 16: system architecture
# Laid out in the style of a staged pipeline diagram: numbered cards left to
# right, a detail panel inside each, a decision branch off the agent, the loop
# back into the environment, and a plain-language strip along the bottom.

clone_slide(3, 16, after=8, title="System Architecture")
s = read(16)
s = "<p:sp>".join(
    [p for i, p in enumerate(s.split("<p:sp>")) if i == 0 or '<p:ph idx="1"/>' not in p]
)

# Stage palettes: (card fill, border, heading text, panel text)
GREY = ("F5F5F7", "D5D5DC", "3A3A44", "5A5A66")
GRN = ("E8F3EC", "8FBFA3", "1A6B3C", "3D6B50")
BLU = ("E6EEF8", "9BB4D6", "1F4E79", "3C5F80")
AMB = ("FDF3E0", "DCBE86", "8A5A00", "6E5220")
PUR = ("F0E9F7", "B9A2D4", "5B3E8E", "5A4A70")
OK_ = ("E8F5E9", "7CB342", "2E7D32", "2E7D32")
HOLD = ("F3F4F6", "B8BDC4", "44484F", "44484F")
FINAL = ("E3F2FD", "7BAEDC", "0D47A1", "1B5E8A")

# Left edge matches the slide title's margin and the right edge matches the
# other slides' content width, so the diagram reads as part of the deck rather
# than a floating image. Card width is derived from that span, not chosen, so
# the row cannot drift off the edge when the stage count changes.
CARD_X0 = X  # 838198, the template's content left edge
CARD_RIGHT = X + CX  # 11353798
N_CARDS = 5
CARD_GAP = 200000
CARD_W = (CARD_RIGHT - CARD_X0 - (N_CARDS - 1) * CARD_GAP) // N_CARDS
CARD_Y = 1520000
CARD_H = 2300000
PITCH = CARD_W + CARD_GAP


def card_x(i: int) -> int:
    return CARD_X0 + i * PITCH


def card_mid(i: int) -> int:
    return card_x(i) + CARD_W // 2


def stage(sid: int, i: int, pal, title, sub: str, panel: list[str]) -> str:
    """One numbered stage: coloured card, heading, and an inset detail panel.

    The panel is a separate shape rather than more paragraphs in the card,
    because a white inset is what separates 'what this stage is' from 'what is
    inside it' at a glance -- the same split the reference diagram uses.
    """
    fill, border, head_col, panel_col = pal
    x = card_x(i)
    # Titles are split across explicit lines rather than left to wrap. A card is
    # 2.27in wide, which holds about 26 characters at 10.5pt, and several of
    # these titles sit within a character or two of that -- close enough that
    # the break would land differently on another machine's font metrics.
    title_lines = [title] if isinstance(title, str) else list(title)
    head = box(
        sid,
        x,
        CARD_Y,
        CARD_W,
        CARD_H,
        [(t, 10.5, True, head_col) for t in title_lines] + [(sub, 8, False, panel_col)],
        fill,
        border,
    )
    # Push the heading to the top of the card so the panel sits below it.
    head = head.replace('<a:bodyPr anchor="ctr"', '<a:bodyPr anchor="t"', 1)
    if not panel:
        return head
    lines = [(t, 7.5, False, panel_col) for t in panel]
    panel_sp = box(
        sid + 1,
        x + 100000,
        CARD_Y + 820000,
        CARD_W - 200000,
        CARD_H - 930000,
        lines,
        "FFFFFF",
        border,
    )
    panel_sp = panel_sp.replace('algn="ctr"', 'algn="l"').replace(
        '<a:bodyPr anchor="ctr"', '<a:bodyPr anchor="ctr"', 1
    )
    return head + panel_sp


shapes = []

shapes.append(
    stage(
        300,
        0,
        GREY,
        ("DAILY STATE", "one day, t of 180"),
        "what the agent sees",
        ["10 products", "3 suppliers", "stock on hand", "orders in transit"],
    )
)
shapes.append(
    stage(
        310,
        1,
        GRN,
        ("1. DEMAND", "CALIBRATION"),
        "UCI Online Retail II",
        [
            "1,067,371 transactions cleaned",
            "Gamma-Poisson demand fitted",
            "weekday and month factors",
            "variance about 216x the mean",
            "fitted before 1 Jun 2011 only",
        ],
    )
)
shapes.append(
    stage(
        320,
        2,
        BLU,
        ("2. SUPPLY CHAIN", "SIMULATOR"),
        "Gymnasium environment",
        [
            "3 suppliers, 1-8 day lead time",
            "80-98% fill rate, MOQ, outages",
            "one shared 20,000-unit store",
            "revenue, holding, ordering,",
            "lost-sale and overflow costs",
        ],
    )
)
shapes.append(
    stage(
        330,
        3,
        AMB,
        ("3. STATE FEATURE", "EXTRACTION"),
        "88 values, rebuilt daily",
        [
            "stock, in transit, days of cover",
            "7-day mean, volatility, trend",
            "unmet backlog per product",
            "weekday and season",
            "supplier lead time and fill",
        ],
    )
)
shapes.append(
    stage(
        340,
        4,
        PUR,
        ("4. MASKABLE PPO", "AGENT"),
        "the learned policy",
        [
            "input: the 88 features",
            "MLP, two 256-unit layers",
            "21 choices per product:",
            "7 quantities x 3 suppliers",
            "illegal choices masked out",
        ],
    )
)

for i in range(4):
    shapes.append(arrow(350 + i, card_x(i) + CARD_W + 20000, 2600000, 160000, 140000, down=False))

# Decision branch off the agent, mirroring the reference's two-way split.
BRANCH_Y = 4050000
BRANCH_H = 520000
BRANCH_GAP = 80000
BRANCH_W = (CARD_W - BRANCH_GAP) // 2
bx0 = card_x(4)
bx1 = bx0 + BRANCH_W + BRANCH_GAP
shapes.append(arrow(360, bx0 + BRANCH_W // 2 - 80000, 3830000, 160000, 200000))
shapes.append(arrow(361, bx1 + BRANCH_W // 2 - 80000, 3830000, 160000, 200000))
shapes.append(
    box(
        362,
        bx0,
        BRANCH_Y,
        BRANCH_W,
        BRANCH_H,
        [("ORDER", 10, True, OK_[2]), ("qty + supplier", 7, False, OK_[3])],
        OK_[0],
        OK_[1],
    )
)
shapes.append(
    box(
        363,
        bx1,
        BRANCH_Y,
        BRANCH_W,
        BRANCH_H,
        [("HOLD", 10, True, HOLD[2]), ("no order today", 7, False, HOLD[3])],
        HOLD[0],
        HOLD[1],
    )
)

# The loop back into the environment. This is the part that makes the problem
# sequential rather than a series of independent forecasts, so it is drawn
# rather than left to the caption.
LOOP_Y = 4290000
LOOP_X = card_mid(2)
shapes.append(box(370, LOOP_X, LOOP_Y, bx0 + 40000 - LOOP_X, 38100, [], "9BB4D6", "9BB4D6"))
shapes.append(arrow(371, LOOP_X - 80000, 3830000, 160000, 470000))
shapes.append(
    textbox(
        372,
        LOOP_X + 80000,
        4340000,
        bx0 - LOOP_X - 160000,
        300000,
        para("action applied, next day begins", 0, 800, False, "3C5F80", False),
    )
)

shapes.append(arrow(380, bx0 + BRANCH_W // 2 - 80000, 4590000, 160000, 400000))
shapes.append(
    box(
        381,
        7450000,
        5000000,
        CARD_RIGHT - 7450000,
        800000,
        [
            ("EPISODE OUTCOME", 10.5, True, FINAL[2]),
            ("180 days of profit, scored on 30 held-out seeds", 8, False, FINAL[3]),
            ("against four grid-search tuned classical policies", 8, False, FINAL[3]),
        ],
        FINAL[0],
        FINAL[1],
    )
)

# Bottom strip: the same flow stated once in words, for a reader who does not
# want to trace arrows.
how = (
    para("HOW IT WORKS", 0, 900, True, "3A3A44", False)
    + para(
        "Real sales history is reduced to demand statistics, which drive the "
        "simulator. Each simulated day the simulator hands the agent 88 "
        "numbers describing stock, demand and supplier health.",
        0,
        800,
        False,
        "44484F",
        False,
    )
    + para(
        "The agent picks a quantity and a supplier for all 10 products at "
        "once; illegal choices are blocked before it chooses. The simulator "
        "applies the order, returns that day's profit, and the loop repeats "
        "for 180 days.",
        0,
        800,
        False,
        "44484F",
        False,
    )
    + para(
        "Next phase: an LLM turns each decision into a business-language "
        "explanation, and generates the disruption scenarios used to stress "
        "test it.",
        0,
        800,
        True,
        "7A5210",
        False,
    )
)
HOW_W = 7450000 - 200000 - CARD_X0
shapes.append(box(390, CARD_X0, 4750000, HOW_W, 1450000, [], "FAFAFC", "E0E0E8"))
shapes.append(textbox(391, CARD_X0 + 160000, 4870000, HOW_W - 320000, 1220000, how))

write(16, add_shapes(s, "".join(shapes)))

# ======================================= slide 17: visual results (two figures)

clone_slide(11, 17, after=11, title="Preliminary Results and Discussion")
s = read(17)
s = "<p:sp>".join(
    [p for i, p in enumerate(s.split("<p:sp>")) if i == 0 or '<p:ph idx="1"/>' not in p]
)
s = re.sub(r"<p:sp><p:nvSpPr><p:cNvPr id=\"4[34]\".*?</p:sp>", "", s, flags=re.S)

FIGDIR = (
    r"D:\SEM-7\Reinforcement leaning\SupplyAI-RL Explainable Supply Chain "
    r"Optimization\results\figures"
)
add_image_rel(17, "rIdFigA", "figure2.png", f"{FIGDIR}\\05_policy_comparison.png")
add_image_rel(17, "rIdFigB", "figure3.png", f"{FIGDIR}\\06_training_curves.png")

FW = 5100000
GAPF = 315600
pics = []
for i, (rid, fname) in enumerate(
    [("rIdFigA", "05_policy_comparison.png"), ("rIdFigB", "06_training_curves.png")]
):
    w, h = png_size(f"{FIGDIR}\\{fname}")
    # Height follows each file's own aspect ratio; forcing a shared height
    # would stretch one of the two charts.
    pics.append(picture(200 + i, rid, X + i * (FW + GAPF), 2050000, FW, round(FW * h / w)))

caps = para(
    "Left: every policy on the 30 held-out reporting seeds. The best "
    "classical policy leads the agent, and the spread across seeds is wide "
    "enough that only a paired test can settle the difference.",
    0,
    1150,
    False,
    INK,
    False,
) + para(
    "Right: learning progress on the training-eval seeds. Action masking "
    "lifts the whole curve; both runs flatten below the classical "
    "reference line, and the unmasked run was still improving at 1M steps, "
    "which is why the sweep trains to 2M.",
    0,
    1150,
    False,
    INK,
    False,
)

s = add_shapes(s, "".join(pics) + textbox(210, X, 4950000, CX, 1200000, caps))
s = add_shapes(
    s,
    textbox(
        211,
        X,
        Y,
        CX,
        400000,
        para(
            "Visual evidence for the numbers on the previous slide, generated "
            "directly from the stored result files.",
            0,
            1400,
            False,
            DARKGREEN,
            False,
        ),
    ),
)
write(17, s)

# ================================================================== package

if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(TPL.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(TPL).as_posix())

print("wrote", OUT.resolve())
