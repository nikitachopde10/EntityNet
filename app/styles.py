"""Inline CSS and small HTML helpers for the EntityNet dashboard."""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "primary": "#6366F1",
    "primary_dark": "#4338CA",
    "ink": "#0F172A",
    "ink_soft": "#475569",
    "muted": "#94A3B8",
    "panel": "#FFFFFF",
    "panel_soft": "#F8FAFC",
    "border": "#E2E8F0",
    "person": "#3B82F6",
    "company": "#10B981",
    "sanction": "#EF4444",
    "news": "#F59E0B",
    "focal": "#6366F1",
}

RISK_COLORS = {
    "none": ("#16A34A", "#DCFCE7", "Clean"),
    "low": ("#16A34A", "#DCFCE7", "Low"),
    "medium": ("#CA8A04", "#FEF9C3", "Medium"),
    "high": ("#EA580C", "#FFEDD5", "High"),
    "severe": ("#B91C1C", "#FECACA", "Severe"),
}

ENTITY_COLORS = {
    "Person": PALETTE["person"],
    "Company": PALETTE["company"],
    "Sanction": PALETTE["sanction"],
    "NewsArticle": PALETTE["news"],
}


CSS = f"""
<style>
:root {{
    --primary: {PALETTE["primary"]};
    --primary-dark: {PALETTE["primary_dark"]};
    --ink: {PALETTE["ink"]};
    --ink-soft: {PALETTE["ink_soft"]};
    --muted: {PALETTE["muted"]};
    --border: {PALETTE["border"]};
    --panel: {PALETTE["panel"]};
    --panel-soft: {PALETTE["panel_soft"]};
}}

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    color: var(--ink);
}}

/* Hide the default Streamlit header chrome */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}}
section[data-testid="stSidebar"] * {{ color: #F1F5F9 !important; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 6px;
    transition: all .15s ease;
}}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.55);
}}

/* Hero / banner */
.en-hero {{
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
    color: #ffffff;
    padding: 28px 32px;
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(99,102,241,0.18);
    margin-bottom: 22px;
}}
.en-hero h1 {{
    margin: 0 0 6px 0;
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}}
.en-hero p {{
    margin: 0;
    color: rgba(255,255,255,0.85);
    font-size: 15px;
    max-width: 880px;
}}
.en-hero .en-chips {{
    margin-top: 14px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}
.en-chip {{
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.28);
    color: #ffffff;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: .01em;
}}

/* Cards */
.en-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    height: 100%;
}}
.en-card-soft {{ background: var(--panel-soft); }}
.en-card h3 {{
    margin: 0 0 4px 0;
    font-size: 13px;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
}}
.en-card .en-value {{
    font-size: 28px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.1;
    margin-top: 4px;
}}
.en-card .en-sub {{
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
}}

/* Risk badge */
.en-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: .01em;
}}

/* Section heading */
.en-section {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 26px 0 12px 0;
}}
.en-section h2 {{
    font-size: 18px;
    font-weight: 700;
    color: var(--ink);
    margin: 0;
}}
.en-section span {{ color: var(--muted); font-size: 13px; }}

/* Pill */
.en-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
}}

/* Path lines */
.en-path {{
    background: var(--panel-soft);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 13px;
    color: var(--ink);
}}
.en-path b {{ color: var(--primary-dark); }}

/* Buttons */
.stButton > button {{
    background: var(--primary);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
}}
.stButton > button:hover {{ background: var(--primary-dark); color: #ffffff; }}

/* Inputs */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{
    border-radius: 10px !important;
}}

/* Dataframe */
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

/* Compare panels */
.en-compare {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    height: 100%;
}}
.en-compare h4 {{
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
}}
.en-compare .en-meta {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
.en-compare ul {{ margin: 8px 0 0 0; padding-left: 16px; }}
.en-compare li {{ font-size: 13px; color: var(--ink); margin-bottom: 3px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        chip_spans = "".join(f'<span class="en-chip">{c}</span>' for c in chips)
        chips_html = f'<div class="en-chips">{chip_spans}</div>'
    st.markdown(
        f"""
        <div class="en-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="en-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="en-card"><h3>{label}</h3>'
        f'<div class="en-value">{value}</div>{sub_html}</div>'
    )


def risk_badge(level: str) -> str:
    color, bg, label = RISK_COLORS.get(level.lower(), ("#475569", "#E2E8F0", level.upper()))
    return (
        f'<span class="en-badge" style="background:{bg};color:{color};">'
        f'● {label.upper()}</span>'
    )


def section(title: str, hint: str = "") -> None:
    hint_html = f"<span>{hint}</span>" if hint else ""
    st.markdown(
        f'<div class="en-section"><h2>{title}</h2>{hint_html}</div>',
        unsafe_allow_html=True,
    )
