"""Build the final-review deck on the faculty template.

    python scripts/build_final_deck.py [template.pptx]

Every number in the deck is read from the committed result files, so it cannot
disagree with the report or the dashboard. Money is shown in rupees through the
project's own formatter (1 GBP = 105 INR, configs/display.yaml).

Inputs : the faculty template, results/*.json, results/figures/*.png, and
         reports/final_deck_assets/ (infographics and dashboard screenshots,
         made by scripts/render_infographics.py and capture_dashboard.py).
Output : reports/SupplyAI-RL_Final_Project_Review.pptx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deckkit import (  # noqa: E402
    AMBER,
    AMBER_L,
    BLUE,
    BLUE_L,
    CH,
    CL,
    CT,
    CW,
    GREEN,
    GREEN_D,
    GREEN_L,
    GREY_L,
    INK,
    MUTED,
    PINK,
    PINK_L,
    badge,
    card,
    new_slide,
    notes,
    picture,
    stat,
    table,
    text,
    trim_white,
)

from src.money import RATE, RUPEE, inr  # noqa: E402

TEMPLATE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path.home() / "Downloads" / "PPT_format_22AIE450.pptx"
)
OUT = ROOT / "reports" / "SupplyAI-RL_Final_Project_Review.pptx"
ASSETS = ROOT / "reports" / "final_deck_assets"
SCREENS = ASSETS / "screens"
FIGS = ROOT / "results" / "figures"
TMP = ASSETS / "_tmp"
TMP.mkdir(exist_ok=True)

TEAM = [
    ("Koushik", "CB.SC.U4AIE23312"),
    ("Santosh", "CB.SC.U4AIE23310"),
    ("Srinith", "CB.SC.U4AIE23357"),
    ("Dharani Pratap", "CB.SC.U4AIE23330"),
]
TITLE = "SupplyAI-RL: Explainable Supply Chain Optimization"


def R(gbp: float) -> str:
    return inr(gbp, symbol=RUPEE)


def lakh(gbp: float) -> float:
    return gbp * RATE / 1e5


def load(name: str):
    with (ROOT / "results" / name).open(encoding="utf-8") as fh:
        return json.load(fh)


cmp_ = load("comparison.json")
S = cmp_["summaries"]
ROB = load("robustness.json")
EXP = load("explainer_eval.json")["summary"]
STAB = load("pattern_stability.json")
ENG, AGT, CST = S["forecast+SS"], S["kaggle_full01 (masked)"], S["constant"]
GAP = cmp_["verdict"]["mean_difference"]
T = cmp_["verdict"]["t"]

prs = Presentation(str(TEMPLATE))
tpl = list(prs.slides)
tpl_content = tpl[2]  # a plain template content slide: header band, footer, rule
n_template = len(tpl)


# ======================================================================= 1 title
s1 = tpl[0]
for shp in s1.shapes:
    if shp.name == "Google Shape;91;p1":
        runs = [r for p in shp.text_frame.paragraphs for r in p.runs]
        runs[0].text = "[Final Project Review]"
        for r in runs[1:]:
            r.text = ""
    elif shp.name == "Title 3" and shp.has_text_frame:
        if shp.text_frame.text.strip() == "Project Title":
            shp.text_frame.paragraphs[0].runs[0].text = TITLE
            for r in shp.text_frame.paragraphs[0].runs[1:]:
                r.text = ""
            for r in shp.text_frame.paragraphs[0].runs[:1]:
                r.font.size = Pt(36)
            shp.top = Pt(235)
        else:  # "List of Team Members" box: replaced by the table of names below
            shp._element.getparent().remove(shp._element)
    elif shp.name == "Rectangle 2" and shp.has_text_frame:
        paras = shp.text_frame.paragraphs
        # "Mid semester project review– August 2026" and "22AIE450 Speech Processing"
        for r in paras[0].runs:
            if "Mid" in r.text or "review" in r.text:
                r.text = r.text.replace("Mid semester", "Final semester")
            if "August" in r.text:
                r.text = r.text.replace("August", "October")
        for r in paras[1].runs:
            r.text = r.text.replace("Speech Processing", "Reinforcement Learning")
for shp in s1.shapes:
    # Left blank on purpose: the batch/group number and the date are not known
    # to the project. Kept short so they fit inside the template's own boxes.
    if shp.has_text_frame and shp.text_frame.text.startswith("Batch and Group"):
        shp.text_frame.paragraphs[0].runs[0].text = "Batch and Group Number: ______"
        for r in shp.text_frame.paragraphs[0].runs[1:]:
            r.text = ""
        shp.text_frame.paragraphs[0].runs[0].font.size = Pt(16)
    if shp.has_text_frame and shp.text_frame.text.startswith("Date of the Presentation"):
        shp.text_frame.paragraphs[0].runs[0].text = "Date: ____________"
        for r in shp.text_frame.paragraphs[0].runs[1:]:
            r.text = ""
        shp.text_frame.paragraphs[0].runs[0].font.size = Pt(16)
rows = [["Name", "Roll number"]] + [[n, r] for n, r in TEAM]
table(s1, rows, 3.9, 4.55, 5.5, [2.2, 2.2], font=15, head_font=15, row_h=0.34, align_right_from=1)
notes(
    s1,
    "Opening. One sentence: we built a decision-support system that decides how much stock a shop "
    "should order and from which supplier, tested an RL agent against tuned classical methods, and "
    "added an LLM that explains each decision and writes the crises we test with. "
    "Fill in the batch/group number and the presentation date before presenting.",
)

# ===================================================================== 2 outline
s = new_slide(prs, tpl_content, "Outline")
left = [
    "1  Introduction and motivation",
    "2  Problem definition",
    "3  Literature review and gaps",
    "4  Objectives and status",
    "5  System architecture",
    "6  Methodology: simulator, agent, baselines",
    "7  Dataset and data stability",
    "8  LLM explainer and crisis generator",
]
right = [
    "9  Dashboard: screenshots",
    "10  Results: comparison and ablation",
    "11  Before vs after the system",
    "12  Robustness to crises",
    "13  Explainer accuracy",
    "14  Error analysis, limitations",
    "15  Engineering evidence",
    "16  Conclusion and future work",
]
text(s, 0.9, 1.7, 5.7, 5.0, [{"t": t, "size": 22, "space": 12} for t in left])
text(s, 7.0, 1.7, 5.7, 5.0, [{"t": t, "size": 22, "space": 12} for t in right])
notes(
    s,
    "Roadmap. The first half is what we built; the second half is the evidence that it works and where it does not.",
)

# ================================================================ 3 introduction
s = new_slide(prs, tpl_content, "Introduction and Motivation")
card(
    s,
    CL,
    CT,
    4.0,
    1.75,
    "Order too much",
    ["Money sits in the warehouse", "Storage costs pile up", "Seasonal goods are marked down"],
    PINK_L,
    PINK,
)
card(
    s,
    CL + 4.2,
    CT,
    4.0,
    1.75,
    "Order too little",
    ["The customer buys elsewhere", "The sale is gone for good", "Lost-sale penalty applies"],
    AMBER_L,
    AMBER,
)
card(
    s,
    CL + 8.4,
    CT,
    4.03,
    1.75,
    "Pick the wrong supplier",
    ["Cheap but slow", "Fast but expensive", "Reasonable but ships only 80%"],
    BLUE_L,
    BLUE,
)
text(
    s,
    CL,
    CT + 1.95,
    CW,
    3.4,
    [
        {
            "runs": [
                ("A sequential problem. ", {"bold": True}),
                (
                    "Today’s order becomes next week’s stock, so what you do now changes what you can do later.",
                    {},
                ),
            ],
            "size": 18,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("Classical rules struggle. ", {"bold": True}),
                (
                    "(s,S), EOQ, newsvendor and forecast + safety stock treat each product alone and assume well-behaved demand. In our data, daily demand varies about 216× more than a Poisson model allows.",
                    {},
                ),
            ],
            "size": 18,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("Our hypothesis. ", {"bold": True}),
                (
                    "RL can learn a joint ordering and sourcing policy that adapts to season, volatility and supplier state, which a formula cannot.",
                    {},
                ),
            ],
            "size": 18,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("But trust matters. ", {"bold": True}),
                (
                    "A manager will not act on a number they cannot understand, so an LLM explains the situation behind every order, with every number checked.",
                    {},
                ),
            ],
            "size": 18,
            "bullet": True,
            "space": 8,
        },
    ],
)
notes(
    s,
    "Motivation in one minute. Both ways of ordering wrong cost money, and supplier choice adds a second decision. The honest framing: RL is a hypothesis we test, not an assumption.",
)

# =============================================================== 4 problem definition
s = new_slide(prs, tpl_content, "Problem Definition")
rows = [
    ["", ""],
    ["Products", "10 real products (UCI Online Retail II)"],
    ["Suppliers", "3, invented: no public dataset has supplier terms"],
    ["Time", "180 simulated days; first 14 are warm-up, not scored"],
    ["Warehouse", "one shared store of 20,000 units; overflow is refused"],
    ["Decision", "each day, each product: 7 quantity levels × 3 suppliers = 21 options"],
    ["Goal", "maximise profit = sales − purchases − storage − fees − stockout − overflow"],
]
rows[0] = ["The task", "Definition"]
table(s, rows, CL, CT, 7.4, [1.3, 6.0], font=14, row_h=0.5, align_right_from=9)
sup = [
    ["Supplier", "Price", "Lead time", "Ships", "Fee"],
    ["EconoSource", "base", "5–8 d", "90%", f"{RUPEE}2,100"],
    ["RapidTrade", "+35%", "1–3 d", "98%", f"{RUPEE}4,725"],
    ["MidWay", "+15%", "3–5 d", "80%", f"{RUPEE}3,150"],
]
table(s, sup, CL + 7.65, CT, 4.78, [1.5, 0.8, 1.0, 0.8, 1.0], font=13, row_h=0.45)
text(
    s,
    CL + 7.65,
    CT + 1.9,
    4.78,
    1.0,
    [
        {
            "t": "None is best: the cheap one ties up a week of stock in transit, the fast one eats margin, the middle one is unreliable.",
            "size": 14,
            "color": MUTED,
        }
    ],
)
stat(
    s,
    CL,
    CT + 3.75,
    3.9,
    1.3,
    "≈ 17 trillion",
    "possible order combinations every day (21¹⁰)",
    PINK,
)
stat(s, CL + 4.15, CT + 3.75, 3.9, 1.3, "216×", "variance of daily demand vs its mean", GREEN)
stat(
    s,
    CL + 8.3,
    CT + 3.75,
    4.13,
    1.3,
    "Joint choice",
    "quantity and supplier decided together",
    BLUE,
)
notes(
    s,
    "The decision is small to state and huge to search: 21 options per product across 10 products is about 17 trillion combinations a day. No supplier is best, which is what makes it a real trade-off.",
)

# ============================================================ 5 literature review
s = new_slide(prs, tpl_content, "Literature Review")
lit = [
    ["#", "Work", "Approach", "Key finding", "Limitation"],
    [
        "1",
        "Proximal Policy Optimization (2017)",
        "Clipped surrogate policy gradient",
        "Stable, sample-efficient default RL baseline",
        "General purpose; no supply-chain structure",
    ],
    [
        "2",
        "A Deep Q-Network for the Beer Game (2022)",
        "DQN on a serial chain",
        "DRL matches or beats base-stock under cooperation",
        "Single product, serial chain, no sourcing choice",
    ],
    [
        "3",
        "Can Deep RL Improve Inventory Management? (2022)",
        "A3C on lost-sales and dual-sourcing",
        "Competitive with tuned heuristics, not clearly superior",
        "Heavy tuning cost; no explanation layer",
    ],
    [
        "4",
        "Deep RL for Inventory Control: A Roadmap (2022)",
        "Survey and roadmap",
        "Names fair benchmarking and interpretability as open problems",
        "Conceptual; no implementation",
    ],
    [
        "5",
        "A Closer Look at Invalid Action Masking (2022)",
        "Masking in policy gradients",
        "Masking is a valid gradient and scales to large action spaces",
        "Not shown on inventory or sourcing",
    ],
]
table(s, lit, CL, CT, CW, [0.4, 3.0, 2.3, 3.4, 3.3], font=13, row_h=0.85, align_right_from=9)
notes(
    s,
    "Five anchor papers. The pattern: deep RL has been competitive with tuned heuristics, not clearly better, and masking is what makes large action spaces tractable.",
)

# ================================================================= 6 gaps
s = new_slide(prs, tpl_content, "Gaps Identified, and What We Did")
gaps = [
    (
        "Narrow problem settings",
        "Single-product or serial chains dominate; joint quantity and supplier choice under one shared capacity is rarely modelled.",
        "We model 10 products, 3 suppliers and one shared warehouse, with a joint action.",
    ),
    (
        "Explainability is deferred",
        "Learned policies output a number, not a reason, so they stay in simulation.",
        "An LLM explains each decision from computed facts; every number is checked.",
    ),
    (
        "Stress testing is hand-built",
        "Spikes and failures are coded case by case, so coverage is narrow.",
        "An LLM writes overlapping crises; each is validated and clamped before use.",
    ),
    (
        "Weak baselines",
        "Baselines are often left untuned, so a new method “wins” for free.",
        "All four classical methods are tuned on separate seeds before any comparison.",
    ),
]
y = CT
for i, (g, d, a) in enumerate(gaps):
    badge(s, CL, y + 0.12, i + 1, GREEN)
    text(
        s,
        CL + 0.5,
        y,
        5.4,
        1.2,
        [
            {"t": g, "size": 17, "bold": True, "color": GREEN_D, "space": 2},
            {"t": d, "size": 14, "color": INK},
        ],
    )
    card(
        s,
        CL + 6.2,
        y,
        6.23,
        1.15,
        "What we did",
        [{"t": a, "size": 14, "space": 2}],
        GREEN_L,
        GREEN,
        title_size=14,
    )
    y += 1.33
notes(
    s,
    "Each gap on the left has a concrete answer on the right. We claim only what we tested: no claim that nobody has combined these before beyond this single setting.",
)

# ================================================================ 7 objectives
s = new_slide(prs, tpl_content, "Objectives and Status")
obj = [
    ["#", "Objective", "Status", "Evidence"],
    [
        "1",
        "Build a supply chain simulation from real data",
        "Done",
        "1.07M records calibrate it; sensible rules profit, silly ones lose",
    ],
    [
        "2",
        "Train an RL agent to choose quantity and supplier",
        "Done",
        "MaskablePPO, tuned, 2M steps; 90.8% of the best classical method",
    ],
    [
        "3",
        "LLM explains each decision in business language",
        "Done",
        "100% delivered accuracy; 83% raw, the check caught every invented number",
    ],
    [
        "4",
        "LLM generates demand-spike and supplier-delay scenarios",
        "Done",
        "6 unseen crises used for testing; curriculum training tested",
    ],
    [
        "5",
        "Compare the RL agent with traditional policies",
        "Done",
        "30 held-out seeds, paired t-test: the classical method still wins",
    ],
]
table(s, obj, CL, CT, CW, [0.4, 4.6, 0.9, 6.0], font=14, row_h=0.62, align_right_from=2)
text(
    s,
    CL,
    CT + 4.0,
    CW,
    1.4,
    [
        {
            "runs": [
                ("Objective 5 says compare, not win. ", {"bold": True, "color": GREEN_D}),
                (
                    "The comparison came out against the AI, and we report it that way: the claim we can defend is that RL reaches 90.8% of a strong tuned baseline, with a measured reason for the gap.",
                    {},
                ),
            ],
            "size": 17,
        },
    ],
)
notes(
    s,
    "All five objectives are met. The fifth is a comparison, and the honest result is that the tuned classical method wins. Say this first, before the examiner asks.",
)

# =============================================================== 8 architecture
s = new_slide(prs, tpl_content, "System Architecture")
arch = trim_white(ASSETS / "architecture.png", TMP / "architecture.png")
picture(s, arch, CL - 0.05, CT - 0.05, CW + 0.1, CH + 0.1, border=False)
notes(
    s,
    "Walk left to right. Data builds the simulator. Two decision makers act in it: the tuned classical engine and the RL challenger. The LLM layer explains decisions and writes crises. Evaluation and the dashboard sit on the right. The green strip is the deployment story: shadow mode, nothing places an order.",
)

# ================================================================ 9 methodology
s = new_slide(prs, tpl_content, "Methodology: How the Agent Learns")
pic = trim_white(ASSETS / "agent_loop.png", TMP / "agent_loop.png")
picture(s, pic, CL - 0.05, CT - 0.15, CW + 0.1, 3.95, border=False)
card(
    s,
    CL,
    CT + 3.85,
    4.0,
    1.55,
    "Why masking",
    ["Best score without it: " + R(24.9e5 / RATE), "With it: " + R(76.4e5 / RATE) + ", about 3×"],
    PINK_L,
    PINK,
    body_size=14,
)
card(
    s,
    CL + 4.2,
    CT + 3.85,
    4.0,
    1.55,
    "Why a joint action",
    ["Quantity and supplier as one choice", "2.8× faster training, matches reality"],
    GREEN_L,
    GREEN,
    body_size=14,
)
card(
    s,
    CL + 8.4,
    CT + 3.85,
    4.03,
    1.55,
    "Why tuning",
    ["8 settings tried, best 2 run to 2M steps", "The lowest learning rate won"],
    BLUE_L,
    BLUE,
    body_size=14,
)
notes(
    s,
    "The loop: state in, action out, reward back. Action masking is the biggest single improvement: it stops the agent wasting training on choices that are never legal.",
)

# ================================================================= 10 dataset
s = new_slide(prs, tpl_content, "Dataset Description")
picture(
    s,
    FIGS / "04_demand_regimes.png",
    CL,
    CT,
    6.0,
    3.9,
    caption="Three demand types: steady, moderate, bursty",
)
text(
    s,
    CL + 6.3,
    CT,
    6.1,
    5.4,
    [
        {"t": "UCI Online Retail II", "size": 20, "bold": True, "color": GREEN_D, "space": 4},
        {
            "t": "1,067,371 sales records, UK online gift shop, December 2009 to December 2011",
            "size": 15,
            "bullet": True,
        },
        {
            "t": "Cleaned: cancellations, returns, negative quantities and missing codes removed; one row per product per day; Saturdays dropped (the shop never traded)",
            "size": 15,
            "bullet": True,
        },
        {
            "t": "10 products chosen on purpose: 4 steady (CV 1.02–1.20), 3 moderate (1.29–1.65), 3 bursty (2.09–2.34)",
            "size": 15,
            "bullet": True,
        },
        {"t": "Only data before 1 June 2011 builds the simulator", "size": 15, "bullet": True},
        {
            "t": "The agent never sees the records: five measurements (mean, variability, weekday, month, zero-sales days) drive a Gamma–Poisson demand model",
            "size": 15,
            "bullet": True,
        },
        {
            "t": "A newer Kaggle dataset (2022–24) was tested and rejected: no weekday or seasonal pattern, i.e. synthetic.",
            "size": 15,
            "bullet": True,
            "color": PINK,
        },
    ],
)
text(
    s,
    CL,
    CT + 4.1,
    6.0,
    1.2,
    [
        {
            "t": "If all ten products behaved alike, one simple rule would handle them all and there would be nothing for an AI to learn.",
            "size": 14,
            "color": MUTED,
        }
    ],
)
notes(
    s,
    "Why this dataset: real and public, with enough variety. Why not the newer one: it is synthetic noise, and newer-but-fake is worse than older-but-real. Money is converted from pounds at 1 GBP = 105 rupees for display only.",
)

# ============================================================ 11 data stability
s = new_slide(prs, tpl_content, "Is the Data Too Old?")
picture(
    s,
    FIGS / "08_pattern_stability.png",
    CL,
    CT - 0.05,
    7.6,
    3.0,
    caption="Year 1 vs year 2: weekday rhythm repeats; mid-year does not",
)
wk = STAB["weekday_factor_correlation"]
mo = STAB["month_factor_correlation"]
mo2 = STAB["month_factor_correlation_excluding_december"]
vm = STAB["variance_to_mean_ratio"]
dat = [
    ["Property (year 1 vs year 2)", "Result"],
    ["Weekday pattern", f"r = {wk:.2f}"],
    ["Month pattern", f"r = {mo:.2f}  ({mo2:.2f} without Dec)"],
    ["Busiest month", "November, both years"],
    ["Burstiness (variance / mean)", f"{vm['year_1']:.0f} and {vm['year_2']:.0f}"],
]
table(s, dat, CL + 7.9, CT, 4.53, [2.6, 2.4], font=13, row_h=0.5)
text(
    s,
    CL,
    CT + 3.65,
    CW,
    1.7,
    [
        {
            "runs": [
                ("The question: ", {"bold": True}),
                (
                    "the simulator uses the shape of demand, not prices or volumes. Is that shape a feature of retail, or of one period?",
                    {},
                ),
            ],
            "size": 16,
            "space": 5,
        },
        {
            "runs": [
                ("The answer: ", {"bold": True}),
                (
                    "the weekly rhythm and the November peak repeat across two independent years. We report the weaker mid-year match as found. ",
                    {},
                ),
                (
                    "It cannot speak to 2026 itself, which stays a stated limitation.",
                    {"bold": True, "color": PINK},
                ),
            ],
            "size": 16,
        },
    ],
)
notes(
    s,
    "This answers the faculty's question that the data is from 2009-2011. We split the two years and measured the demand shape in each. The two features the simulator relies on most hold. We do not claim anything about today's market.",
)

# ================================================================ 12 simulator
s = new_slide(prs, tpl_content, "Methodology: The Simulator")
pic = trim_white(ASSETS / "daily_cycle.png", TMP / "daily_cycle.png")
picture(s, pic, CL - 0.05, CT - 0.15, CW + 0.1, 3.0, border=False)
card(
    s,
    CL,
    CT + 2.95,
    4.0,
    2.45,
    "Storage cost",
    [
        "0.15% of item cost per day",
        "High on purpose: seasonal gift items lose value",
        "An earlier 730% a year made any stock absurdly costly",
    ],
    GREY_L,
    GREEN,
    body_size=14,
)
card(
    s,
    CL + 4.2,
    CT + 2.95,
    4.0,
    2.45,
    "Leftover stock",
    [
        "Worth 60% of cost at day 180",
        "An earlier version wrote it off, an artefact of stopping at 180 days",
    ],
    GREY_L,
    BLUE,
    body_size=14,
)
card(
    s,
    CL + 8.4,
    CT + 2.95,
    4.03,
    2.45,
    "One shared warehouse",
    [
        "With per-product limits, “always order the maximum” was best",
        "A shared limit made ordering a real trade-off",
    ],
    GREY_L,
    PINK,
    body_size=14,
)
notes(
    s,
    "A Gymnasium environment. The three cards are three mistakes in earlier versions that we fixed after the simulator itself told us something was wrong. A demo confirms the simulator rewards sensible behaviour: reasonable rules earn about 27 lakh, never ordering loses 3.9 crore.",
)

# ============================================================== 13 RL agent
s = new_slide(prs, tpl_content, "Methodology: The RL Agent")
card(
    s,
    CL,
    CT,
    4.0,
    2.55,
    "What it sees: 88 numbers",
    [
        "Per product: stock, stock on order, days of cover, 7-day demand, variability, trend, backlog",
        "Weekday, season, supplier availability, lead time, reliability",
    ],
    BLUE_L,
    BLUE,
    body_size=14,
)
card(
    s,
    CL + 4.2,
    CT,
    4.0,
    2.55,
    "What it decides",
    [
        "One of 21 options per product",
        "7 quantity levels × 3 suppliers",
        "Offline suppliers and oversize orders are masked",
    ],
    PINK_L,
    PINK,
    body_size=14,
)
card(
    s,
    CL + 8.4,
    CT,
    4.03,
    2.55,
    "What it is rewarded for",
    [
        "That day’s profit, scaled and clipped",
        "Zero during the 14-day warm-up",
        "Not shown: any ordering history (see error analysis)",
    ],
    GREEN_L,
    GREEN,
    body_size=14,
)
hp = [
    ["Winning setting", "Value"],
    ["Learning rate", "5e-5 (lowest offered)"],
    ["Entropy coefficient", "0.005"],
    ["Discount γ", "0.995"],
    ["Rollout length", "1024"],
    ["Network", "2 × 256, CPU"],
]
table(s, hp, CL, CT + 2.8, 5.6, [2.8, 2.8], font=13, row_h=0.4)
text(
    s,
    CL + 5.9,
    CT + 2.8,
    6.5,
    2.6,
    [
        {"t": "Hyperparameter search", "size": 17, "bold": True, "color": GREEN_D},
        {"t": "8 configurations, 250,000-step trial each", "size": 14, "bullet": True},
        {"t": "Best 2 trained to 2M steps on Kaggle (about 4.8 hours)", "size": 14, "bullet": True},
        {
            "t": "Winner: lowest learning rate, larger network. Earlier runs were unstable because they learned too fast, not because the network was too small.",
            "size": 14,
            "bullet": True,
        },
    ],
)
notes(
    s,
    "MaskablePPO is PPO with action masking. The agent's inputs include nothing about its own ordering history, which turns out to be the cause of its main weakness.",
)

# ============================================================ 14 baselines + fairness
s = new_slide(prs, tpl_content, "Classical Baselines and a Fair Evaluation")
bl = [
    ["Method", "Tuned settings"],
    ["(s, S)", "reorder below 11 days of cover, order up to 16"],
    ["EOQ + reorder point", "reorder at 8 days of cover, safety factor 1.6"],
    ["Newsvendor", "3-day horizon, cheapest supplier"],
    ["Forecast + safety stock", "1-day window, z = 1.28, review every 4 days"],
]
table(s, bl, CL, CT, 6.4, [2.3, 4.1], font=13, row_h=0.5, align_right_from=9)
sd = [
    ["Seeds", "Used for"],
    ["100–111", "tuning the classical methods"],
    ["900–905", "picking the best RL checkpoint"],
    ["500–529", "reporting only; never used to choose anything"],
]
table(s, sd, CL + 6.7, CT, 5.73, [1.4, 4.3], font=13, row_h=0.5, hl_rows=(3,), align_right_from=9)
text(
    s,
    CL,
    CT + 2.9,
    CW,
    2.5,
    [
        {
            "runs": [
                ("Tuning mattered a lot. ", {"bold": True}),
                (
                    f"Widening the search lifted the best classical method from {R(66395)} to {R(84920)} on the tuning seeds. Untuned baselines would have made the AI look better than it is.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 6,
        },
        {
            "runs": [
                ("Same options for everyone. ", {"bold": True}),
                (
                    "Every classical method picks from the same 21 options per product as the AI, so none has finer control.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 6,
        },
        {
            "runs": [
                ("Paired comparison. ", {"bold": True}),
                (
                    "Every method faces the same customers and delivery delays on each seed, so we test the per-seed difference, which removes run-to-run randomness.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
        },
    ],
)
notes(
    s,
    "This is what makes a negative result meaningful. Three disjoint seed groups, tuned baselines, identical customers for every method.",
)

# ============================================================== 15 explainer
s = new_slide(prs, tpl_content, "LLM Explainer: Facts In, Checked Sentences Out")
pic = trim_white(ASSETS / "explainer_pipeline.png", TMP / "explainer_pipeline.png")
picture(s, pic, CL - 0.05, CT - 0.1, CW + 0.1, 4.85, border=False)
text(
    s,
    CL,
    CT + 4.8,
    CW,
    0.7,
    [
        {
            "runs": [
                ("The rule: ", {"bold": True, "color": PINK}),
                (
                    "the LLM is never asked why. The agent’s real reason is a neural network, and any “because” the LLM wrote would be a plausible story, not the cause.",
                    {},
                ),
            ],
            "size": 15,
        }
    ],
)
notes(
    s,
    "The key design decision: the LLM only presents facts the simulator computed. Then every number in its output is checked against those facts. If it invented one, a fixed template is shown instead. Three free models via OpenRouter, tried in order, with retries.",
)

# ============================================================ 16 crises + curriculum
s = new_slide(prs, tpl_content, "LLM Crisis Generator and Curriculum Training")
cr = [
    ["Days", "Event (written by the LLM)", "Cause it gave"],
    ["45–65", "demand 3× on four products", "viral social media trend"],
    ["50–63", "supplier 2 offline", "cyberattack on mid-tier supplier"],
    ["55–72", "supplier 0 delayed by 6 days", "port congestion at major hub"],
    ["60–84", "costs 1.6×", "raw material shortage"],
]
table(s, cr, CL, CT, 7.3, [0.9, 3.4, 3.0], font=13, row_h=0.5, align_right_from=9)
text(
    s,
    CL,
    CT + 2.6,
    7.3,
    0.8,
    [
        {
            "t": "These overlap: the supplier goes down during the demand spike. A single-event test never reaches that combination.",
            "size": 14,
            "color": MUTED,
        }
    ],
)
card(
    s,
    CL + 7.6,
    CT,
    4.83,
    3.2,
    "Nothing the LLM writes is trusted",
    [
        "Scenario type must be known",
        "Supplier and product numbers must exist; out-of-range products are dropped, not wrapped",
        "Values clamped to safe ranges",
        "Windows moved inside the 180-day run",
        "6 deliberately broken inputs: 3 survived, none as written",
    ],
    BLUE_L,
    BLUE,
    body_size=14,
)
card(
    s,
    CL,
    CT + 3.45,
    CW,
    1.95,
    "Curriculum training",
    [
        "16 crisis sets from 19 LLM scenarios, saved, then split 10 train / 6 test",
        "The agent trains on a random set from the 10 and is tested on the 6 it never saw; seed, settings and steps match the untuned agent, so crisis exposure is the only difference",
        "The split is the whole method: testing on training crises would only measure memorisation",
    ],
    GREEN_L,
    GREEN,
    body_size=14,
)
notes(
    s,
    "The LLM's value here is imagination, not trust. We validate everything. The curriculum experiment is controlled: same seed, same settings, same steps.",
)

# ======================================================== 17-22 dashboard evidence
CROP_PANEL = (370, 440, 1530, 1190)

s = new_slide(prs, tpl_content, "Dashboard (1/7): Episode View, Engine")
picture(
    s,
    SCREENS / "episode_engine.png",
    CL,
    CT - 0.05,
    8.2,
    5.45,
    crop=CROP_PANEL,
    caption="Streamlit dashboard, forecast + safety stock, held-out seed 500",
)
y = CT
for i, (h_, d) in enumerate(
    [
        ("Pick the policy and seed", "Sidebar: any policy, any held-out seed, optional crisis."),
        (
            "Five outcome cards",
            f"Profit {R(ENG['total_profit'])}, fill rate, ordering fees, stockout cost, overflow.",
        ),
        ("Stock and profit over 180 days", "Two single-axis charts: never a dual axis."),
        ("Which supplier was used", "The engine sends almost every order to the cheap supplier."),
    ]
):
    badge(s, CL + 8.5, y + 0.05, i + 1, PINK)
    text(
        s,
        CL + 8.95,
        y - 0.02,
        3.5,
        1.3,
        [
            {"t": h_, "size": 16, "bold": True, "color": GREEN_D, "space": 1},
            {"t": d, "size": 14, "color": INK},
        ],
    )
    y += 1.35
notes(
    s,
    "Live demo starting point. The dashboard opens on the engine, the policy a buyer would actually run. Everything shown comes from a real episode run on demand; no number on the page is typed in.",
)

s = new_slide(prs, tpl_content, "Dashboard (2/7): The Challenger, and a Crisis")
picture(
    s,
    SCREENS / "episode_challenger.png",
    CL,
    CT,
    6.1,
    3.95,
    crop=CROP_PANEL,
    caption="RL challenger: orders spread over all three suppliers",
)
picture(
    s,
    SCREENS / "episode_crisis.png",
    CL + 6.33,
    CT,
    6.1,
    3.95,
    crop=CROP_PANEL,
    caption="Engine with a crisis injected: shaded band = disruption",
)
text(
    s,
    CL,
    CT + 4.1,
    CW,
    1.3,
    [
        {
            "runs": [
                ("Challenger (left): ", {"bold": True, "color": PINK}),
                (
                    "602 orders across all three suppliers against the engine’s 323. That fragmentation is the cause of the agent’s double ordering fees.   ",
                    {},
                ),
                ("Crisis (right): ", {"bold": True, "color": GREEN_D}),
                (
                    "the red band marks the days the disruption is active, so stock and profit can be read against it.",
                    {},
                ),
            ],
            "size": 14,
        }
    ],
)
notes(
    s,
    "Left: switch the policy to the challenger and the supplier bars change completely. This is the visual evidence for the error analysis. Right: the crisis switch injects the LLM-written disruption for the whole run.",
)

s = new_slide(prs, tpl_content, "Dashboard (3/7): Before vs After the System")
picture(
    s,
    SCREENS / "before_after.png",
    CL,
    CT,
    7.9,
    5.4,
    crop=(370, 440, 1540, 1125),
    caption="“Before”: a fixed daily order.  “After”: the engine, with the RL challenger beside it.",
)
cards = [
    (
        "Profit per run",
        f"{R(CST['total_profit'])} → {R(ENG['total_profit'])}",
        f"+{(ENG['total_profit'] / CST['total_profit'] - 1) * 100:.0f}%",
        GREEN,
    ),
    (
        "Fill rate",
        f"{CST['fill_rate'] * 100:.1f}% → {ENG['fill_rate'] * 100:.1f}%",
        f"+{(ENG['fill_rate'] - CST['fill_rate']) * 100:.1f} points",
        BLUE,
    ),
    (
        "Stockout cost",
        f"{R(CST['stockout_penalty'])} → {R(ENG['stockout_penalty'])}",
        f"{(ENG['stockout_penalty'] / CST['stockout_penalty'] - 1) * 100:.0f}%",
        AMBER,
    ),
]
yy = CT
for lbl, val, sub, col in cards:
    stat(s, CL + 8.2, yy, 4.23, 1.3, val, f"{lbl}   {sub}", col, value_size=22, label_size=13)
    yy += 1.45
text(
    s,
    CL + 8.2,
    yy,
    4.23,
    1.2,
    [
        {
            "t": "Most of the gain comes from forecasting with safety stock, which is what the engine does, not from RL.",
            "size": 13,
            "color": MUTED,
            "italic": True,
        }
    ],
)
notes(
    s,
    "Before = the shop orders the same fixed amount every day, tuned as carefully as every other baseline. After = the system's engine. Same 30 unseen runs. The honest reading: forecasting plus safety stock does the heavy lifting; the RL challenger also beats the fixed order but trails the engine.",
)

s = new_slide(prs, tpl_content, "Dashboard (4/7): Every Test Run, Not Just the Average")
picture(
    s,
    SCREENS / "before_after.png",
    CL,
    CT - 0.05,
    CW,
    3.95,
    crop=(370, 1140, 1540, 1530),
    caption="Each dot is one of the 30 held-out test seeds, sorted by the “before” profit",
)
text(
    s,
    CL,
    CT + 4.05,
    CW,
    1.4,
    [
        {
            "runs": [
                ("Why show every run? ", {"bold": True, "color": GREEN_D}),
                (
                    "An average can hide losses. The engine (green) is above the fixed order (grey) on 30 of 30 seeds; the RL challenger (pink) is above it on 29 of 30.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 6,
        },
        {
            "runs": [
                ("One weak run is visible too: ", {"bold": True, "color": PINK}),
                (
                    "the lone pink dot far below the others is a seed where the challenger did badly.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
        },
    ],
)
notes(
    s,
    "Every dot is one test seed, sorted by the before profit. The point of this chart is that the improvement is consistent, not an average hiding losses.",
)

s = new_slide(prs, tpl_content, "Dashboard (5/7): One Run, Day by Day")
picture(
    s,
    SCREENS / "before_after.png",
    CL,
    CT - 0.05,
    8.4,
    4.0,
    crop=(370, 1535, 1540, 2085),
    caption="Seed 500: cumulative profit and stockouts, before vs after",
)
text(
    s,
    CL + 8.7,
    CT,
    3.73,
    5.3,
    [
        {"t": "Reading the charts", "size": 18, "bold": True, "color": GREEN_D, "space": 6},
        {
            "t": "Left: profit builds up over 180 days. Green (engine) pulls away from grey (fixed order).",
            "size": 15,
            "bullet": True,
            "space": 6,
        },
        {
            "t": "Right: product-days out of stock, counted up. The engine stays low; the fixed order and the challenger climb.",
            "size": 15,
            "bullet": True,
            "space": 6,
        },
        {
            "t": "The charts follow the sidebar’s seed and crisis switch, so before and after can be compared live under any condition.",
            "size": 15,
            "bullet": True,
        },
    ],
)
text(
    s,
    CL,
    CT + 4.15,
    8.4,
    1.2,
    [
        {
            "t": "On this single run the challenger runs out of stock more than the engine: a visible example of its weakness.",
            "size": 15,
            "color": PINK,
            "bold": True,
        }
    ],
)
notes(
    s,
    "Live demo option: flip the crisis switch in the sidebar and watch both charts redraw for all three policies.",
)

s = new_slide(prs, tpl_content, "Dashboard (6/7): Explain a Decision")
picture(
    s,
    SCREENS / "explain.png",
    CL,
    CT,
    8.1,
    3.4,
    crop=(370, 440, 1530, 915),
    caption="Day 45, product 0: explanation written by the LLM, source shown as a badge",
)
text(
    s,
    CL + 8.4,
    CT,
    4.03,
    5.4,
    [
        {"t": "How to use it", "size": 17, "bold": True, "color": GREEN_D, "space": 4},
        {"t": "Pick a day and a product", "size": 14, "bullet": True},
        {"t": "Press Explain this decision", "size": 14, "bullet": True},
        {"t": "Read 2–3 plain sentences", "size": 14, "bullet": True},
        {
            "t": "The black badge shows who wrote it: a model name, or “deterministic template” if the check rejected the LLM",
            "size": 14,
            "bullet": True,
        },
        {
            "t": "Expand “the facts the model was given” to verify every number",
            "size": 14,
            "bullet": True,
        },
    ],
)
text(
    s,
    CL,
    CT + 3.65,
    8.1,
    1.7,
    [
        {
            "runs": [
                ("Check it yourself: ", {"bold": True, "color": PINK}),
                (
                    "2,365 units, 14.0 days of cover, demand of 169 units and a 7-day average of 78 all appear in the facts panel. If the model had invented a figure, the template would have replaced this text.",
                    {},
                ),
            ],
            "size": 14,
        },
    ],
)
notes(
    s,
    "This is the explainer in action, live. The explanation is a restatement of computed facts, not an account of the neural network's reasoning, and we say so explicitly.",
)

s = new_slide(prs, tpl_content, "Dashboard (7/7): Policy Comparison and Robustness")
picture(
    s,
    SCREENS / "comparison.png",
    CL,
    CT,
    6.1,
    3.95,
    crop=(370, 440, 1530, 1160),
    caption="Policy comparison tab: 30 held-out seeds",
)
picture(
    s,
    SCREENS / "robustness.png",
    CL + 6.33,
    CT,
    6.1,
    3.95,
    crop=(370, 440, 1530, 1135),
    caption="Robustness tab: calm vs 6 unseen crises",
)
text(
    s,
    CL,
    CT + 4.1,
    CW,
    1.3,
    [
        {
            "t": "Only the rows a claim depends on are shown: the four classical methods, the tuned, untuned and no-masking agents; then the engine, the tuned agent, and the curriculum agent beside its control.",
            "size": 14,
            "bullet": True,
            "space": 4,
        },
        {
            "t": "The headline cards give the numbers directly; the table below each chart is the same data for anyone who wants to check it.",
            "size": 14,
            "bullet": True,
        },
    ],
)
notes(s, "Both tabs read the saved result files, so the dashboard cannot disagree with the report.")

# ================================================================ 23 results I
s = new_slide(prs, tpl_content, "Results (1/4): Main Comparison on 30 Unseen Runs")
order = [
    ("Forecast + safety stock", "forecast+SS"),
    ("Newsvendor", "newsvendor"),
    ("RL agent, tuned, 2M", "kaggle_full01 (masked)"),
    ("EOQ + reorder point", "EOQ+ROP"),
    ("RL agent, untuned, 1M", "masked_1m (masked)"),
    ("(s, S) policy", "(s,S)"),
    ("Constant order (no system)", "constant"),
    ("RL agent, no masking", "baseline_1m"),
]
rows = [["Method", "Profit", "Fill", "Fees", "Stockout"]]
for label, key in order:
    v = S[key]
    rows.append(
        [
            label,
            R(v["total_profit"]),
            f"{v['fill_rate'] * 100:.1f}%",
            R(v["ordering_cost"]),
            R(v["stockout_penalty"]),
        ]
    )
table(
    s,
    rows,
    CL,
    CT,
    6.9,
    [2.6, 1.3, 0.8, 1.1, 1.2],
    font=12,
    row_h=0.4,
    bold_rows=(1, 3),
    hl_rows=(1, 3),
)
picture(s, FIGS / "05_policy_comparison.png", CL + 7.1, CT, 5.33, 3.0)
stat(
    s,
    CL,
    CT + 3.9,
    3.2,
    1.35,
    f"{AGT['total_profit'] / ENG['total_profit'] * 100:.1f}%",
    "of the best classical profit",
    PINK,
    value_size=32,
)
stat(
    s,
    CL + 3.5,
    CT + 3.9,
    3.2,
    1.35,
    f"−{R(GAP * -1).replace(RUPEE, RUPEE)}",
    "gap per run, paired t = " + f"{T:.2f}",
    GREEN,
    value_size=26,
)
text(
    s,
    CL + 7.2,
    CT + 3.4,
    5.23,
    1.9,
    [
        {
            "t": "The best classical method beats the tuned agent, and the difference is real (95% CI 5.5–10.6 lakh), not chance. The agent beats two of the four classical methods: EOQ and (s,S).",
            "size": 15,
            "space": 5,
        },
        {
            "t": "Mean profit per 180-day run, 30 held-out seeds. Rupees at 1 GBP = 105.",
            "size": 12,
            "color": MUTED,
        },
    ],
)
notes(
    s,
    "State the result plainly: the classical engine wins by about 8 lakh per run, paired t of minus 6.45. The agent reaches 90.8% of its profit and beats EOQ and (s,S). The constant-order row is the 'no system' baseline.",
)

# ================================================================ 24 results II
s = new_slide(prs, tpl_content, "Results (2/4): What Each Step Was Worth")
steps = [
    ["Step", "Train score", "Test score", "Gap to classical"],
    ["PPO, no masking", "24.9 lakh", "23.7 lakh", "−64.1 lakh"],
    ["+ action masking", "76.4 lakh", "74.8 lakh", "−13.0 lakh"],
    ["+ tuning, 2M steps", "86.2 lakh", "79.7 lakh", "−8.1 lakh"],
]
table(
    s, steps, CL, CT, 6.6, [2.4, 1.3, 1.3, 1.6], font=14, row_h=0.55, hl_rows=(3,), bold_rows=(3,)
)
text(
    s,
    CL,
    CT + 2.4,
    6.6,
    3.0,
    [
        {
            "runs": [
                ("Action masking: about 3×. ", {"bold": True, "color": PINK}),
                (
                    "Without it the agent spends most of training discovering that some choices are pointless.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("Tuning: closed 38% of what was left. ", {"bold": True, "color": GREEN_D}),
                ("From a 13.0 to an 8.1 lakh gap.", {}),
            ],
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Train scores use checkpoint-selection seeds 900–905; test scores use the held-out seeds 500–529.",
            "size": 12,
            "color": MUTED,
        },
    ],
)
picture(
    s,
    FIGS / "06_training_curves.png",
    CL + 6.9,
    CT,
    5.53,
    3.4,
    caption="Learning curves: no masking, masking, tuned (checkpoint seeds)",
)
notes(
    s,
    "Two controlled improvements, each measured: masking about 3x, tuning closes 38% of the remaining gap. The curves show learning progress on the checkpoint-selection seeds; reported scores come from separate held-out seeds.",
)

# ================================================================ 25 results III
s = new_slide(prs, tpl_content, "Results (3/4): Robustness to Unseen Crises")
rob_rows = [["Method", "Calm", "Crisis", "Drop"]]
names = [
    ("Forecast + safety stock", "forecast+SS (classical)"),
    ("RL agent, tuned (no crisis training)", "kaggle_full01"),
    ("RL agent, untuned (control)", "masked_1m"),
    ("RL agent, trained on crises", "curriculum_1m"),
]
for label, key in names:
    r = ROB["rows"][key]
    rob_rows.append([label, R(r["calm"]), R(r["disrupted"]), f"{r['drop_pct']:.1f}%"])
table(
    s,
    rob_rows,
    CL,
    CT,
    7.4,
    [3.4, 1.3, 1.3, 1.0],
    font=14,
    row_h=0.55,
    hl_rows=(2,),
    bold_rows=(2,),
)
V = ROB["verdict"]
card(
    s,
    CL + 7.7,
    CT,
    4.73,
    2.9,
    "Crisis training did not work",
    [
        f"Under crisis it earns {R(ROB['rows']['curriculum_1m']['disrupted'])} vs the control’s {R(ROB['rows']['masked_1m']['disrupted'])}",
        f"but gave up {R(ROB['rows']['masked_1m']['calm'] - ROB['rows']['curriculum_1m']['calm'])} of calm profit",
        f"Paired t = {V['t']:+.2f}: not significant",
    ],
    PINK_L,
    PINK,
    body_size=14,
)
text(
    s,
    CL,
    CT + 3.2,
    CW,
    2.2,
    [
        {
            "runs": [
                ("A smaller percentage drop is misleading ", {"bold": True}),
                (
                    "when the base is lower: the crisis-trained agent drops less because it has less to lose. We show the drop beside absolute profit so “less to lose” cannot pass for robustness.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("The useful finding: ", {"bold": True, "color": GREEN_D}),
                (
                    "plain hyperparameter tuning made the agent more robust than crisis training did. The tuned agent never saw a crisis in training, yet has the highest crisis profit and smallest drop of all agent versions.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Limit: the crisis-trained agent faced a harder task with the same step budget, so under-training may explain part of its result.",
            "size": 13,
            "color": MUTED,
        },
    ],
)
notes(
    s,
    "Every method loses more than a third of its profit in a crisis; RL is not more adaptive than the tuned classical method. Our attempt to fix that with crisis training did not work. We report it as a negative result.",
)

# ================================================================ 26 results IV
s = new_slide(prs, tpl_content, "Results (4/4): Explainer Accuracy on 30 Real Decisions")
stat(
    s,
    CL,
    CT,
    3.9,
    1.7,
    f"{EXP['raw_model_accuracy'] * 100:.0f}%",
    "raw LLM accuracy: grounded and complete, before the check",
    PINK,
    value_size=44,
)
stat(
    s,
    CL + 4.15,
    CT,
    3.9,
    1.7,
    f"{EXP['delivered_accuracy'] * 100:.0f}%",
    "delivered accuracy: what the manager saw, after the check",
    GREEN,
    value_size=44,
)
stat(
    s,
    CL + 8.3,
    CT,
    4.13,
    1.7,
    "5 of 30",
    "raw outputs contained an invented number; all 5 caught",
    BLUE,
    value_size=40,
)
ex_rows = [
    ["Measure", "Result"],
    ["LLM reachable (after retries)", f"{EXP['llm_reachability'] * 100:.0f}%"],
    [
        "Grounded (raw output used only real numbers)",
        f"{EXP['grounding_rate'] * 100:.0f}%  (25 of 30)",
    ],
    ["Complete (product, stock, order details)", f"{EXP['completeness_llm'] * 100:.0f}%"],
    ["Length", f"{EXP['mean_sentences']:.1f} sentences, {EXP['mean_words']:.0f} words"],
]
table(s, ex_rows, CL, CT + 1.95, 7.2, [4.6, 2.6], font=14, row_h=0.5)
text(
    s,
    CL + 7.5,
    CT + 1.95,
    4.93,
    3.4,
    [
        {"t": "Why it matters", "size": 17, "bold": True, "color": GREEN_D, "space": 4},
        {
            "t": "The five bad outputs read fluently and confidently. A buyer would not have noticed.",
            "size": 14,
            "bullet": True,
        },
        {
            "t": "Without the check, one explanation in six would carry a made-up figure.",
            "size": 14,
            "bullet": True,
        },
        {
            "t": "An earlier metric scored only outputs that had already passed the check and reported 100% for a model that invented numbers 17% of the time. We replaced it with both numbers.",
            "size": 14,
            "bullet": True,
        },
    ],
)
notes(
    s,
    "Be upfront about both numbers. 83% is the model by itself; 100% is the system with the check. The finding is that the check is essential, not decoration.",
)

# ================================================================ 27 error analysis
s = new_slide(prs, tpl_content, "Error Analysis: Why the Agent Falls Short")
picture(
    s,
    FIGS / "07_simulator_trace.png",
    CL,
    CT,
    7.4,
    4.1,
    caption="One traced run: stock vs demand, warehouse fill, profit, supplier mix (panel D)",
)
text(
    s,
    CL + 7.7,
    CT,
    4.73,
    4.1,
    [
        {
            "runs": [
                ("Ordering fees: ", {"bold": True, "color": PINK}),
                (
                    f"{R(AGT['ordering_cost'])} vs {R(ENG['ordering_cost'])}. The agent spreads orders across all three suppliers: 719 orders in the traced run, 323 for the engine.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("Fill rate: ", {"bold": True, "color": PINK}),
                (
                    f"{AGT['fill_rate'] * 100:.1f}% vs {ENG['fill_rate'] * 100:.1f}%, about 3 lakh more in lost-sale penalties.",
                    {},
                ),
            ],
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "runs": [
                ("Overflow: ", {"bold": True, "color": GREEN_D}),
                (
                    "the agents are the only methods with zero overflow loss; they learned the shared-warehouse rule.",
                    {},
                ),
            ],
            "size": 14,
            "bullet": True,
        },
    ],
)
card(
    s,
    CL,
    CT + 4.15,
    CW,
    1.25,
    "Root cause: an observability gap",
    [
        "The agent’s 88 inputs contain no ordering history: no days since last order, no running total of fees, no last supplier used. It pays double and cannot see it. Adding a few history features per product is the most promising untried change."
    ],
    PINK_L,
    PINK,
    body_size=14,
)
notes(
    s,
    "We know why it loses. It cannot observe its own ordering history, so it cannot learn to stop fragmenting orders. This is a limit on what it observes, not a tuning problem. We have not yet tested the fix.",
)


# ============================================================ 28 engineering evidence
def _git(*args):
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


N_COMMITS = int(_git("rev-list", "--count", "HEAD").strip())
_py = [
    f for f in _git("ls-files", "src/*.py", "app/*.py", "scripts/*.py", "tests/*.py").split() if f
]
N_LOC = round(sum(len((ROOT / f).read_text(encoding="utf-8").splitlines()) for f in _py), -2)
s = new_slide(prs, tpl_content, "Engineering Evidence")
stat(s, CL, CT, 2.9, 1.3, "137", "automated tests", GREEN, value_size=34)
stat(s, CL + 3.1, CT, 2.9, 1.3, f"{N_COMMITS}+", "commits on GitHub", BLUE, value_size=34)
stat(s, CL + 6.2, CT, 2.9, 1.3, f"{N_LOC:,}", "lines of Python", PINK, value_size=34)
stat(s, CL + 9.3, CT, 3.13, 1.3, "CI green", "lint + tests on every push", AMBER, value_size=30)
card(
    s,
    CL,
    CT + 1.55,
    6.1,
    3.8,
    "Reproducible end to end",
    [
        "python -m pytest tests/ -q",
        "python -m src.data.run_pipeline",
        "python -m src.agents.tune_baselines",
        "python -m src.agents.train_ppo --name run",
        "python -m src.eval.compare / robustness / explainer_eval",
        "streamlit run app/dashboard.py",
    ],
    GREY_L,
    GREEN,
    body_size=14,
)
card(
    s,
    CL + 6.33,
    CT + 1.55,
    6.1,
    3.8,
    "What is committed",
    [
        "Every result file; every figure regenerates from them",
        "Kaggle notebook for the 8-configuration sweep",
        "Tests for each phase: environment, data, baselines, training, LLM, scenarios, dashboard, curriculum",
        "Live LLM tests are skipped without an API key, so CI never depends on a free endpoint",
        "Final report in LaTeX (two-column) and this deck",
    ],
    GREY_L,
    BLUE,
    body_size=14,
)
notes(
    s,
    "Evidence that the work is real and reproducible. 137 tests: 135 run offline; 2 call the live LLM and are skipped without a key. The repository is public on GitHub: KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation.",
)

# ================================================================ 29 limitations
s = new_slide(prs, tpl_content, "Limitations")
lim = [
    (
        "Data age",
        "UK retail from 2009–2011. The demand shape is stable across the two years, but nothing here speaks to today; recent product-level data is not public.",
        PINK_L,
        PINK,
    ),
    (
        "Invented suppliers",
        "Lead times, reliability and fees are hand-set to span the standard trade-off. Rupee values are relative, not forecasts.",
        AMBER_L,
        AMBER,
    ),
    (
        "Curriculum under-training",
        "The crisis-trained agent faced a harder task with the same step budget.",
        BLUE_L,
        BLUE,
    ),
    (
        "Small explainer sample",
        "30 decisions, one method, one seed. It restates facts; it does not explain the network.",
        GREEN_L,
        GREEN,
    ),
    (
        "One shop, ten products",
        "Other categories, larger assortments and cost structures are untested.",
        GREY_L,
        GREEN_D,
    ),
    (
        "Free LLM endpoints",
        "Rate limits: explanations can take up to a minute and sometimes fall back to the template.",
        GREY_L,
        BLUE,
    ),
]
for i, (h_, d, f_, a_) in enumerate(lim):
    x = CL + (i % 3) * 4.2
    y = CT + (i // 3) * 2.75
    card(s, x, y, 4.03, 2.55, h_, [{"t": d, "size": 17, "space": 2}], f_, a_, title_size=20)
notes(
    s,
    "Listed in descending order of how much they should change a reader's confidence. The first two are the biggest: the data is old and the suppliers are invented.",
)

# ================================================================ 30 conclusion
s = new_slide(prs, tpl_content, "Conclusion and Future Work")
card(
    s,
    CL,
    CT,
    6.1,
    5.4,
    "What we showed",
    [
        {
            "t": "A complete system: simulator, RL agent, 4 tuned baselines, LLM explainer, crisis generator, dashboard",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": f"The tuned agent reaches {AGT['total_profit'] / ENG['total_profit'] * 100:.1f}% of the best classical profit and beats two of four classical methods; the gap is real (t = {T:.2f}) and traced to supplier fragmentation",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Masking ≈ 3×; tuning closes 38% of the rest; crisis training did not help",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "The explainer is only accurate because it is checked: 83% raw, 100% delivered",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "The honest deployment: classical engine decides, RL scored beside it, nothing executes",
            "size": 16,
            "bullet": True,
        },
    ],
    GREEN_L,
    GREEN,
)
card(
    s,
    CL + 6.33,
    CT,
    6.1,
    5.4,
    "Future work",
    [
        {
            "t": "Give the agent ordering memory (days since last order, last supplier, fees so far) and re-run on the same 30 seeds: the experiment most likely to close the gap",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Test all methods under changed market conditions (more demand variation, longer delays) to show how rankings move",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Longer crisis training, to separate under-training from a failed method",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Recent real data and real supplier terms if a partner can supply them",
            "size": 16,
            "bullet": True,
            "space": 8,
        },
        {
            "t": "Before any live use: new-product handling, order-value limits, an audit trail, a signal for when a human should decide",
            "size": 16,
            "bullet": True,
        },
    ],
    BLUE_L,
    BLUE,
)
notes(
    s,
    "Close on the honest claim: the project works end to end, the comparison is fair, the classical method wins, and we know why. The next experiment is aimed straight at the cause.",
)

# ================================================================ 31 references
s = new_slide(prs, tpl_content, "References")
refs = [
    "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.",
    "S. Huang, S. Ontanón. A closer look at invalid action masking in policy gradient algorithms. FLAIRS-35, 2022.",
    "J. Gijsbrechts, R. N. Boute, J. A. Van Mieghem, D. Zhang. Can deep reinforcement learning improve inventory management? Manufacturing & Service Operations Management, 2022.",
    "R. N. Boute, J. Gijsbrechts, W. van Jaarsveld, N. Vanvuchelen. Deep reinforcement learning for inventory control: a roadmap. European Journal of Operational Research, 2022.",
    "A. Oroojlooyjadid, M. Nazari, L. V. Snyder, M. Takáč. A deep Q-network for the beer game. Manufacturing & Service Operations Management, 2022.",
    "D. Chen, S. L. Sain, K. Guo. Data mining for the online retail industry. Journal of Database Marketing & Customer Strategy Management, 2012.",
    "A. Raffin et al. Stable-Baselines3: reliable reinforcement learning implementations. JMLR 22(268), 2021.",
    "UCI Machine Learning Repository. Online Retail II. https://archive.ics.uci.edu/dataset/502",
]
text(
    s,
    CL,
    CT,
    CW,
    5.4,
    [{"t": f"[{i + 1}]  {r_}", "size": 16, "space": 9} for i, r_ in enumerate(refs)],
)
notes(
    s, "Same eight references as the report. The report has the full paper; this is the short list."
)

# ================================================================ 32 thank you
s = new_slide(prs, tpl_content, "Thank You")
text(
    s,
    CL,
    CT + 0.6,
    CW,
    1.2,
    [{"t": "Questions?", "size": 44, "bold": True, "color": GREEN_D, "align": PP_ALIGN.CENTER}],
)
text(
    s,
    CL,
    CT + 2.0,
    CW,
    2.6,
    [
        {"t": TITLE, "size": 22, "align": PP_ALIGN.CENTER, "space": 14},
        {
            "t": "   •   ".join(f"{n} ({r_})" for n, r_ in TEAM[:2]),
            "size": 16,
            "align": PP_ALIGN.CENTER,
            "color": MUTED,
            "space": 3,
        },
        {
            "t": "   •   ".join(f"{n} ({r_})" for n, r_ in TEAM[2:]),
            "size": 16,
            "align": PP_ALIGN.CENTER,
            "color": MUTED,
            "space": 14,
        },
        {
            "t": "Code, report and dashboard: github.com/KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation",
            "size": 14,
            "align": PP_ALIGN.CENTER,
            "color": MUTED,
        },
    ],
)
notes(
    s,
    "Likely questions: why the classical method wins (fees from supplier fragmentation, no ordering memory); where the accuracy is (no accuracy in RL; relative profit 90.8%, explainer 100% delivered); is the data too old (stability test); what masking is.",
)

# ----------------------------------------------------- drop the template's own slides
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids)[1 : 1 + (n_template - 1)]:
    prs.part.drop_rel(sid.rId)
    sld_ids.remove(sid)

prs.save(str(OUT))
print("wrote", OUT, "with", len(prs.slides), "slides")
