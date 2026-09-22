"""Visual theme for the dashboard: the Clay design system, applied to Streamlit.

Streamlit's own theme (``.streamlit/config.toml``) carries the tokens it can
express: cream canvas, near-black primary, Inter, 12px radius, hairline
borders. This module carries the rest -- pill tabs, the saturated feature
cards, the quote card, the cream footer -- and one Plotly layout so every
chart on the page reads as part of the same system.

Every colour here is a design-system token. Chart marks are the one place
the brand values are re-stepped: the brand teal is too dark and the pastels
too light to serve as data colours on the cream surface, so ``CHART_SERIES``
holds steps of the same hues moved into the legal lightness band. That set
was validated (adjacent-pair colour-vision separation >= 8 and contrast
>= 3:1 against the canvas), not eyeballed.

The system's 3D claymation illustrations are commissioned assets and are not
substituted with flat art, so the hero holds a content card instead.
"""

from __future__ import annotations

import html

import streamlit as st

# ---- colour tokens ----------------------------------------------------------

CANVAS = "#fffaf0"
SURFACE_SOFT = "#faf5e8"
SURFACE_CARD = "#f5f0e0"
SURFACE_STRONG = "#ebe6d6"
INK = "#0a0a0a"
BODY = "#3a3a3a"
MUTED = "#6a6a6a"
MUTED_SOFT = "#9a9a9a"
HAIRLINE = "#e5e5e5"
ON_DARK = "#ffffff"
CORAL = "#ff6b5a"
STATUS = {"success": "#22c55e", "warning": "#f59e0b", "error": "#ef4444"}

# Feature-card surfaces, in the order the system cycles them. Text flips to
# white on the two saturated/dark cards and stays ink on the lighter ones.
CARD_ORDER = ("pink", "teal", "lavender", "peach", "ochre", "cream")
CARD_FILL = {
    "pink": "#ff4d8b",
    "teal": "#1a3a3a",
    "lavender": "#b8a4ed",
    "peach": "#ffb084",
    "ochre": "#e8b94a",
    "cream": SURFACE_CARD,
}
CARD_TEXT = {
    "pink": ON_DARK,
    "teal": ON_DARK,
    "lavender": INK,
    "peach": INK,
    "ochre": INK,
    "cream": INK,
}

# Chart series (fixed order, never cycled): teal-green, pink, lavender, ochre.
CHART_SERIES = ("#007557", "#ff4d8b", "#6b529f", "#98751a")
CLASSICAL = CHART_SERIES[0]  # the engine, everywhere it appears
AGENT = CHART_SERIES[1]  # the challenger, everywhere it appears
DEEMPHASIS = MUTED_SOFT  # context series next to the one that matters
CONTEXT = MUTED  # thin context marks with nothing else to read them by

FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# ---- css ----------------------------------------------------------------------

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* display type: weight 500 with negative tracking; never bolder */
h1 {{ letter-spacing: -1px !important; }}
h2 {{ letter-spacing: -0.5px !important; }}
h3 {{ letter-spacing: -0.3px !important; }}

/* category-tab / category-tab-active: pills, no underline bar */
div[data-testid="stTabs"] [role="tablist"] {{ gap: 8px; border-bottom: 0 !important; box-shadow: none; }}
div[data-testid="stTabs"] [role="tablist"]::after {{ display: none; }}
div[data-testid="stTabs"] [data-testid="stTab"] {{
  border-radius: 9999px; padding: 8px 16px; height: auto; border: 0;
  background: transparent; color: {MUTED}; font-size: 14px; font-weight: 500;
}}
div[data-testid="stTabs"] [data-testid="stTab"] p {{ color: inherit; font-weight: inherit; font-size: 14px; }}
div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
  background: {SURFACE_CARD}; color: {INK};
}}
div[data-testid="stTabs"] .react-aria-SelectionIndicator,
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

/* button-primary / button-secondary: 44px, Inter 14/600 */
div[data-testid="stButton"] button {{
  min-height: 44px; padding: 12px 20px; font-size: 14px; font-weight: 600; border-radius: 12px;
}}

/* feature cards */
.sai-card {{
  border-radius: 24px; padding: 28px 32px; min-height: 172px;
  display: flex; flex-direction: column; justify-content: space-between; gap: 12px;
}}
.sai-card__label {{
  font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;
  line-height: 1.4; opacity: .85;
}}
.sai-card__value {{ font-size: clamp(20px, 1.75vw, 32px); font-weight: 500; letter-spacing: -0.5px; line-height: 1.15; white-space: nowrap; }}
.sai-card__note {{ font-size: 13px; font-weight: 500; line-height: 1.4; opacity: .8; }}

/* testimonial-card, reused for the explanation */
.sai-quote {{ background: {SURFACE_CARD}; border-radius: 16px; padding: 24px; color: {INK}; }}
.sai-quote__text {{ font-size: 16px; line-height: 1.55; margin: 12px 0 0; }}
.sai-quote__note {{ font-size: 13px; font-weight: 500; color: {MUTED}; margin-top: 12px; }}

/* badge-pill */
.sai-badge {{
  display: inline-block; background: {SURFACE_CARD}; color: {INK}; border-radius: 9999px;
  padding: 4px 12px; font-size: 13px; font-weight: 500; line-height: 1.4;
}}
.sai-badge--uppercase {{ font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; }}
.sai-badge--dark {{ background: {INK}; color: {ON_DARK}; }}

/* hero-band + hero-illustration-card */
.sai-hero {{ padding: 8px 0 16px; }}
.sai-hero h1 {{ font-size: 40px; font-weight: 500; line-height: 1.1; letter-spacing: -1px; margin: 12px 0 12px; color: {INK}; }}
.sai-hero p {{ font-size: 16px; line-height: 1.55; color: {BODY}; margin: 0; max-width: 60ch; }}
.sai-hero-card {{ background: {SURFACE_SOFT}; border-radius: 24px; padding: 32px; color: {INK}; }}
.sai-hero-card__title {{ font-size: 18px; font-weight: 600; line-height: 1.4; margin-bottom: 16px; }}
.sai-hero-card__row {{ display: flex; gap: 12px; align-items: baseline; padding: 8px 0; border-top: 1px solid {HAIRLINE}; }}
.sai-hero-card__row:last-child {{ padding-bottom: 0; }}
.sai-hero-card__key {{ flex: 0 0 96px; font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: {MUTED}; }}
.sai-hero-card__val {{ font-size: 14px; line-height: 1.55; color: {BODY}; }}

/* footer: cream, never dark */
.sai-footer {{
  background: {SURFACE_SOFT}; border-radius: 24px; padding: 48px; margin-top: 48px;
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; color: {BODY}; font-size: 14px; line-height: 1.55;
}}
.sai-footer__head {{ font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: {MUTED}; margin-bottom: 8px; }}
@media (max-width: 768px) {{
  .sai-footer {{ grid-template-columns: 1fr; padding: 32px; }}
  .sai-card {{ padding: 24px; min-height: 0; }}
  .sai-hero h1 {{ font-size: 32px; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---- components ---------------------------------------------------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def card_html(label: str, value: str, note: str = "", variant: str = "pink") -> str:
    """One saturated feature card. ``variant`` is a key of CARD_FILL."""
    if variant not in CARD_FILL:
        raise ValueError(f"unknown card variant {variant!r}; use one of {CARD_ORDER}")
    note_html = f'<div class="sai-card__note">{_e(note)}</div>' if note else ""
    return (
        f'<div class="sai-card" style="background:{CARD_FILL[variant]};color:{CARD_TEXT[variant]}">'
        f'<div class="sai-card__label">{_e(label)}</div>'
        f'<div><div class="sai-card__value">{_e(value)}</div>{note_html}</div>'
        "</div>"
    )


def next_variant(previous: str | None) -> str:
    """The card colour that follows ``previous`` in the system's cycle."""
    if previous is None:
        return CARD_ORDER[0]
    return CARD_ORDER[(CARD_ORDER.index(previous) + 1) % len(CARD_ORDER)]


def card_row(items: list[tuple]) -> None:
    """A KPI row of feature cards.

    Each item is ``(label, value, note)`` or ``(label, value, note, variant)``.
    An item without a variant takes the next colour in the cycle, so a row
    never repeats a colour twice in a row unless the caller insists.
    """
    cols = st.columns(len(items))
    previous: str | None = None
    for col, item in zip(cols, items, strict=True):
        label, value, note = item[0], item[1], item[2]
        variant = item[3] if len(item) > 3 and item[3] else next_variant(previous)
        col.markdown(card_html(label, value, note, variant), unsafe_allow_html=True)
        previous = variant


def badge(text: str, uppercase: bool = False, dark: bool = False) -> str:
    cls = "sai-badge"
    if uppercase:
        cls += " sai-badge--uppercase"
    if dark:
        cls += " sai-badge--dark"
    return f'<span class="{cls}">{_e(text)}</span>'


def quote_card(text: str, badge_text: str, note: str = "") -> None:
    """The explanation, on a cream quote card with its source as a badge."""
    note_html = f'<div class="sai-quote__note">{_e(note)}</div>' if note else ""
    st.markdown(
        f'<div class="sai-quote">{badge(badge_text, dark=True)}'
        f'<p class="sai-quote__text">{_e(text)}</p>{note_html}</div>',
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, sub: str, aside_title: str, aside: list[tuple[str, str]]) -> None:
    """The 7/5 hero: headline on the left, a soft content card on the right."""
    left, right = st.columns([7, 5], gap="large")
    left.markdown(
        f'<div class="sai-hero">{badge(kicker, uppercase=True)}'
        f"<h1>{_e(title)}</h1><p>{_e(sub)}</p></div>",
        unsafe_allow_html=True,
    )
    rows = "".join(
        f'<div class="sai-hero-card__row"><div class="sai-hero-card__key">{_e(k)}</div>'
        f'<div class="sai-hero-card__val">{_e(v)}</div></div>'
        for k, v in aside
    )
    right.markdown(
        f'<div class="sai-hero-card"><div class="sai-hero-card__title">{_e(aside_title)}</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def footer(columns: list[tuple[str, list[str]]]) -> None:
    """Cream footer band, three columns of short lines."""
    cols = "".join(
        f'<div><div class="sai-footer__head">{_e(head)}</div>'
        + "".join(f"<div>{_e(line)}</div>" for line in lines)
        + "</div>"
        for head, lines in columns
    )
    st.markdown(f'<div class="sai-footer">{cols}</div>', unsafe_allow_html=True)


# ---- plotly -------------------------------------------------------------------


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def plotly_layout(**overrides) -> dict:
    """Layout shared by every chart: transparent on the canvas, Inter, hairline
    grid one step off the surface, legend above the plot, no chart chrome."""
    axis = {
        "gridcolor": SURFACE_STRONG,
        "gridwidth": 1,
        "zerolinecolor": SURFACE_STRONG,
        "zerolinewidth": 1,
        "linecolor": HAIRLINE,
        "showline": False,
        "ticks": "",
        "automargin": True,
        "tickfont": {"color": MUTED, "size": 12},
        "title": {"font": {"color": MUTED, "size": 12}},
    }
    base = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT_STACK, "color": BODY, "size": 13},
        "title": {"font": {"size": 16, "color": INK}, "x": 0, "xanchor": "left"},
        "margin": {"l": 8, "r": 8, "t": 48, "b": 8, "pad": 4},
        "xaxis": axis,
        "yaxis": axis,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 1.0,
            "xanchor": "right",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": BODY, "size": 12},
        },
        "hoverlabel": {
            "bgcolor": CANVAS,
            "bordercolor": HAIRLINE,
            "font": {"family": FONT_STACK, "color": INK, "size": 13},
        },
        "colorway": list(CHART_SERIES),
        "bargap": 0.45,
        "bargroupgap": 0.08,
    }
    return _merge(base, overrides)


def bar_marker(color: str) -> dict:
    """Bars: rounded 4px at the data end, square at the baseline."""
    return {"color": color, "cornerradius": 4}


def line(color: str) -> dict:
    return {"color": color, "width": 2, "shape": "linear"}


def marker(color: str, symbol: str = "circle", size: int = 9) -> dict:
    """Markers carry a 2px ring in the surface colour so they stay legible
    where they cross a line."""
    return {"color": color, "symbol": symbol, "size": size, "line": {"color": CANVAS, "width": 2}}
