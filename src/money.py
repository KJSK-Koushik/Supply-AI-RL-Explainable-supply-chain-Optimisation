"""Show money in rupees. The simulator works in pounds.

The dataset is a UK retailer, so every price and every profit the simulator
produces is in GBP. Rather than pretend otherwise, values are converted at the
moment they are shown, using one rate stated in configs/display.yaml. Every
result file stays in GBP; only what people read changes.

Indian grouping is used because that is what the reader expects to see:
87,80,520 rather than 8,780,520, and "87.8 lakh" for anything large enough
that the full figure stops being readable at a glance.
"""

from __future__ import annotations

import sys

from src.config import load_config

_CFG = load_config("display")
RATE = float(_CFG["gbp_to_inr"])
CURRENCY = str(_CFG["currency"])
RUPEE = str(_CFG["symbol"])  # always the real sign: figures, dashboard, Markdown


def _console_safe_symbol() -> str:
    """The rupee sign, unless this terminal cannot print it.

    Windows consoles often run cp1252, which has no U+20B9. A demo script
    that crashes on its own currency symbol is worse than one that prints
    'Rs.', so the fallback is chosen once here rather than guarded everywhere.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        RUPEE.encode(enc)
        return RUPEE
    except (UnicodeEncodeError, LookupError):
        return "Rs."


SYMBOL = _console_safe_symbol()


def to_inr(gbp: float) -> float:
    return float(gbp) * RATE


def group_indian(n: float) -> str:
    """12345678 -> '1,23,45,678'. Negative values keep their sign."""
    sign = "-" if n < 0 else ""
    s = str(int(round(abs(n))))
    if len(s) <= 3:
        return sign + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return sign + ",".join(parts) + "," + tail


def inr(gbp: float, short: bool = True, symbol: str | None = None) -> str:
    """Format a GBP amount as rupees.

    short=True gives '₹87.8 lakh' / '₹3.9 crore' for large values, which is
    what a table or a chart label needs. short=False gives the full grouped
    figure, '₹87,80,520', for places where exactness matters more than glance.
    Pass symbol=RUPEE from code that writes to a UTF-8 surface.
    """
    sym = SYMBOL if symbol is None else symbol
    v = to_inr(gbp)
    if short:
        a = abs(v)
        sign = "-" if v < 0 else ""
        if a >= 1e7:
            return f"{sign}{sym}{a / 1e7:.2f} crore"
        if a >= 1e5:
            return f"{sign}{sym}{a / 1e5:.1f} lakh"
    sign = "-" if v < 0 else ""
    return f"{sign}{sym}{group_indian(abs(v))}"


def axis_label(what: str = "profit") -> str:
    return f"{what} ({CURRENCY})"


def lakh_axis(axis) -> None:
    """Label a matplotlib axis in lakh rupees instead of scientific notation.

    An axis reading 0.8 x 1e7 is exactly the kind of thing a reader skips over;
    "80 lakh" is not. Values on the axis are already in rupees.
    """
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1e5:.0f} lakh"))
