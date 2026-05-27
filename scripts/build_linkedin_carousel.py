"""Render the EntityNet LinkedIn carousel.

Outputs:
- assets/linkedin/entitynet_carousel.pdf   ← upload to LinkedIn as a document
- assets/linkedin/slide_3_bar_chart.png    ← standalone bar chart (single-image post)
- assets/linkedin/slide_1_cover.png        ← cover slide PNG
- assets/linkedin/slide_2_table.png        ← headline-table PNG
- assets/linkedin/slide_4_architecture.png ← architecture-diagram PNG
- assets/linkedin/slide_5_cta.png          ← CTA PNG

Run from the project root:
    .venv/bin/python scripts/build_linkedin_carousel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "benchmark_results.json"
OUT_DIR = ROOT / "assets" / "linkedin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1080x1080 square at 144 dpi = 7.5 x 7.5 inches
W = 7.5
H = 7.5
DPI = 144

# Palette
BG = "#FAFAF9"
INK = "#0F172A"
SUB = "#475569"
MUTED = "#94A3B8"
VECTOR = "#94A3B8"
GRAPH = "#6D28D9"
HYBRID = "#0EA5E9"
HIGHLIGHT = "#DC2626"
RULE = "#E2E8F0"


def new_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def footer(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.06,
        0.045,
        "EntityNet · GraphRAG for Adverse Media & Entity Risk",
        fontsize=9,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )
    ax.text(
        0.94,
        0.045,
        label,
        fontsize=9,
        color=SUB,
        ha="right",
        va="center",
        family="DejaVu Sans",
    )
    ax.plot([0.06, 0.94], [0.085, 0.085], color=RULE, lw=0.8)


def load_results() -> dict:
    with RESULTS.open() as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Slide 1 — Cover / hook
# ─────────────────────────────────────────────────────────────────────────────
def slide_1_cover(results: dict) -> plt.Figure:
    fig, ax = new_fig()

    ax.text(
        0.06,
        0.92,
        "ENTITYNET",
        fontsize=11,
        color=GRAPH,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.88,
        "GraphRAG for Adverse Media & Entity Risk",
        fontsize=11,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )

    ax.text(
        0.06,
        0.75,
        "GraphRAG beats",
        fontsize=32,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.68,
        "vector RAG by ~2×",
        fontsize=32,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.61,
        "on entity-risk questions.",
        fontsize=32,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )

    ax.text(
        0.06,
        0.50,
        "50-question public benchmark.",
        fontsize=15,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.455,
        "Reproduces in 90 seconds. No API keys.",
        fontsize=15,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )

    v = results["overall"]["vector"]["f1_mean"]
    g = results["overall"]["graph"]["f1_mean"]
    h = results["overall"]["hybrid"]["f1_mean"]

    box_y = 0.16
    box_h = 0.16

    for i, (label, val, color) in enumerate(
        [
            ("Vector RAG", v, VECTOR),
            ("GraphRAG", g, GRAPH),
            ("Hybrid", h, HYBRID),
        ]
    ):
        x0 = 0.06 + i * 0.30
        rect = mpatches.FancyBboxPatch(
            (x0, box_y),
            0.27,
            box_h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=RULE,
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.135,
            box_y + box_h - 0.035,
            label,
            fontsize=10,
            color=SUB,
            ha="center",
            va="center",
            family="DejaVu Sans",
        )
        ax.text(
            x0 + 0.135,
            box_y + 0.065,
            f"{val:.3f}",
            fontsize=28,
            color=color,
            ha="center",
            va="center",
            weight="bold",
            family="DejaVu Sans",
        )
        ax.text(
            x0 + 0.135,
            box_y + 0.022,
            "F1 (entity-level)",
            fontsize=8,
            color=MUTED,
            ha="center",
            va="center",
            family="DejaVu Sans",
        )

    footer(ax, "1 / 5  ·  swipe →")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 2 — Headline table
# ─────────────────────────────────────────────────────────────────────────────
def slide_2_table(results: dict) -> plt.Figure:
    fig, ax = new_fig()

    ax.text(
        0.06,
        0.92,
        "THE NUMBERS",
        fontsize=11,
        color=GRAPH,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.85,
        "Three retrievers, 50 questions,",
        fontsize=24,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.79,
        "graded by entity-level F1.",
        fontsize=24,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )

    headers = ["Retriever", "F1", "Precision", "Recall", "p50 latency"]
    rows = [
        (
            "Vector RAG (baseline)",
            results["overall"]["vector"]["f1_mean"],
            results["overall"]["vector"]["precision_mean"],
            results["overall"]["vector"]["recall_mean"],
            f"{results['overall']['vector']['latency_ms_p50']:.0f} ms",
            VECTOR,
            False,
        ),
        (
            "GraphRAG",
            results["overall"]["graph"]["f1_mean"],
            results["overall"]["graph"]["precision_mean"],
            results["overall"]["graph"]["recall_mean"],
            f"{results['overall']['graph']['latency_ms_p50']:.0f} ms",
            GRAPH,
            True,
        ),
        (
            "Hybrid (rank-fusion)",
            results["overall"]["hybrid"]["f1_mean"],
            results["overall"]["hybrid"]["precision_mean"],
            results["overall"]["hybrid"]["recall_mean"],
            f"{results['overall']['hybrid']['latency_ms_p50']:.0f} ms",
            HYBRID,
            False,
        ),
    ]

    x_cols = [0.06, 0.45, 0.59, 0.73, 0.88]
    y_top = 0.66
    row_h = 0.10

    for i, header in enumerate(headers):
        ax.text(
            x_cols[i],
            y_top,
            header,
            fontsize=10.5,
            color=SUB,
            ha="left" if i == 0 else "center",
            va="center",
            weight="bold",
            family="DejaVu Sans",
        )
    ax.plot([0.06, 0.94], [y_top - 0.04, y_top - 0.04], color=INK, lw=1.2)

    for j, (name, f1, prec, rec, lat, color, is_winner) in enumerate(rows):
        y = y_top - 0.07 - j * row_h

        if is_winner:
            rect = mpatches.FancyBboxPatch(
                (0.04, y - 0.04),
                0.92,
                0.085,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=0,
                facecolor="#F3E8FF",
            )
            ax.add_patch(rect)

        ax.text(
            x_cols[0],
            y,
            name,
            fontsize=12,
            color=color if is_winner else INK,
            ha="left",
            va="center",
            weight="bold" if is_winner else "normal",
            family="DejaVu Sans",
        )
        for i, val in enumerate([f1, prec, rec]):
            ax.text(
                x_cols[i + 1],
                y,
                f"{val:.3f}",
                fontsize=13,
                color=color if is_winner else INK,
                ha="center",
                va="center",
                weight="bold" if is_winner else "normal",
                family="DejaVu Sans",
            )
        ax.text(
            x_cols[4],
            y,
            lat,
            fontsize=12,
            color=color if is_winner else SUB,
            ha="center",
            va="center",
            weight="bold" if is_winner else "normal",
            family="DejaVu Sans",
        )

    ax.text(
        0.06,
        0.20,
        "GraphRAG nearly doubles vector-RAG F1.",
        fontsize=14,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.16,
        "Multi-hop questions are where similarity",
        fontsize=12,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.13,
        "search runs out of road.",
        fontsize=12,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )

    footer(ax, "2 / 5")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 3 — Per-category bar chart (the headline visual)
# ─────────────────────────────────────────────────────────────────────────────
def slide_3_bar_chart(results: dict) -> plt.Figure:
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)

    header_ax = fig.add_axes([0, 0.78, 1, 0.22])
    header_ax.set_facecolor(BG)
    header_ax.axis("off")
    header_ax.text(
        0.06,
        0.78,
        "THE BREAKDOWN",
        fontsize=11,
        color=GRAPH,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    header_ax.text(
        0.06,
        0.50,
        "F1 by question category",
        fontsize=24,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    header_ax.text(
        0.06,
        0.22,
        "Vector search loses every category — including adverse media.",
        fontsize=12,
        color=SUB,
        ha="left",
        va="center",
        family="DejaVu Sans",
    )

    chart_ax = fig.add_axes([0.10, 0.18, 0.85, 0.55])
    chart_ax.set_facecolor(BG)

    categories = [
        ("direct_sanctions", "Direct\nsanctions"),
        ("ownership_chain", "Ownership\nchain"),
        ("hidden_connection", "Hidden\nconnection"),
        ("risk_propagation", "Risk\npropagation"),
        ("adverse_media", "Adverse\nmedia"),
    ]
    by_cat = results["by_category"]

    import numpy as np

    x = np.arange(len(categories))
    width = 0.26

    vector_vals = [by_cat[c]["vector"]["f1_mean"] for c, _ in categories]
    graph_vals = [by_cat[c]["graph"]["f1_mean"] for c, _ in categories]
    hybrid_vals = [by_cat[c]["hybrid"]["f1_mean"] for c, _ in categories]

    b1 = chart_ax.bar(x - width, vector_vals, width, color=VECTOR, label="Vector RAG", zorder=3)
    b2 = chart_ax.bar(x, graph_vals, width, color=GRAPH, label="GraphRAG", zorder=3)
    b3 = chart_ax.bar(x + width, hybrid_vals, width, color=HYBRID, label="Hybrid", zorder=3)

    chart_ax.set_xticks(x)
    chart_ax.set_xticklabels([lbl for _, lbl in categories], fontsize=10, color=INK)
    chart_ax.set_ylim(0, 1.0)
    chart_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    chart_ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color=SUB)
    chart_ax.set_ylabel("F1 (entity-level)", fontsize=10, color=SUB)
    chart_ax.tick_params(axis="x", length=0, pad=8)
    chart_ax.tick_params(axis="y", colors=SUB, length=0)
    chart_ax.yaxis.grid(True, color=RULE, lw=0.8, zorder=1)
    chart_ax.xaxis.grid(False)
    for spine in ["top", "right", "left", "bottom"]:
        chart_ax.spines[spine].set_visible(False)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            chart_ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.018,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )

    am_vec = by_cat["adverse_media"]["vector"]["f1_mean"]
    am_graph = by_cat["adverse_media"]["graph"]["f1_mean"]
    multiple = am_graph / am_vec
    chart_ax.annotate(
        f"{multiple:.1f}× win",
        xy=(4, am_graph),
        xytext=(4.2, am_graph + 0.10),
        fontsize=11,
        color=HIGHLIGHT,
        weight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=1.2, connectionstyle="arc3,rad=-0.2"),
    )

    legend_ax = fig.add_axes([0, 0.08, 1, 0.06])
    legend_ax.set_facecolor(BG)
    legend_ax.axis("off")
    for i, (label, color) in enumerate(
        [("Vector RAG", VECTOR), ("GraphRAG", GRAPH), ("Hybrid", HYBRID)]
    ):
        x_pos = 0.20 + i * 0.22
        legend_ax.add_patch(mpatches.Rectangle((x_pos, 0.45), 0.022, 0.30, color=color))
        legend_ax.text(
            x_pos + 0.030,
            0.60,
            label,
            fontsize=11,
            color=INK,
            ha="left",
            va="center",
            family="DejaVu Sans",
        )

    foot_ax = fig.add_axes([0, 0, 1, 0.08])
    foot_ax.set_facecolor(BG)
    foot_ax.axis("off")
    footer(foot_ax, "3 / 5  ·  the money slide")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 4 — Architecture
# ─────────────────────────────────────────────────────────────────────────────
def slide_4_architecture() -> plt.Figure:
    fig, ax = new_fig()

    ax.text(
        0.06,
        0.92,
        "HOW IT WORKS",
        fontsize=11,
        color=GRAPH,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.06,
        0.85,
        "Knowledge graph + vectors, then route by intent.",
        fontsize=15,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
        family="DejaVu Sans",
    )

    boxes = [
        (0.05, 0.55, 0.28, 0.18, "Ingest", "Sanctions · Companies\nPersons · News\n\nspaCy NER + fuzzy linker\n→ canonical entity IDs", "#EEF2FF", "#4338CA"),
        (0.36, 0.55, 0.28, 0.18, "Store", "Kuzu graph DB\n(typed nodes + edges)\n\nChromaDB vector index\n(local CPU embeddings)", "#F0FDFA", "#0F766E"),
        (0.67, 0.55, 0.28, 0.18, "Retrieve", "Vector · Graph · Hybrid\n\nIntent → 1 of 7 audited\nCypher traversals", "#FAF5FF", "#6D28D9"),
    ]

    for x, y, w, h, title, body, fc, ec in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.025, title, fontsize=12, color=ec, ha="center", va="center", weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.018, body, fontsize=8.5, color=INK, ha="center", va="center")

    for x_start in [0.33, 0.64]:
        ax.annotate(
            "",
            xy=(x_start + 0.030, 0.64),
            xytext=(x_start, 0.64),
            arrowprops=dict(arrowstyle="->", color=SUB, lw=1.5),
        )

    ax.text(
        0.06,
        0.45,
        "Typed edges = precise traversal",
        fontsize=12,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
    )
    edges = [
        ("OWNS", "Person→Company, Company→Company"),
        ("DIRECTOR_OF", "Person→Company"),
        ("BUSINESS_PARTNER_OF", "Person↔Person, Company↔Company"),
        ("RELATIVE_OF", "Person↔Person"),
        ("SANCTIONS_TARGET", "Sanction→Person/Company"),
        ("MENTIONED_IN", "Person/Company→NewsArticle"),
    ]
    for i, (edge, sig) in enumerate(edges):
        y = 0.40 - i * 0.038
        ax.text(0.07, y, edge, fontsize=9.5, color=GRAPH, ha="left", va="center", weight="bold", family="DejaVu Sans Mono")
        ax.text(0.40, y, sig, fontsize=9.5, color=SUB, ha="left", va="center")

    ax.text(
        0.06,
        0.155,
        "Cypher pattern: (a)-[:OWNS*1..3]->(b) walks ownership only.",
        fontsize=10,
        color=INK,
        ha="left",
        va="center",
        style="italic",
    )
    ax.text(
        0.06,
        0.125,
        "Vector similarity has no concept of OWNS. That's the gap.",
        fontsize=10,
        color=SUB,
        ha="left",
        va="center",
        style="italic",
    )

    footer(ax, "4 / 5")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 5 — Open-source release / CTA
# ─────────────────────────────────────────────────────────────────────────────
def slide_5_cta() -> plt.Figure:
    fig, ax = new_fig()

    ax.text(
        0.06,
        0.92,
        "OPEN SOURCE",
        fontsize=11,
        color=GRAPH,
        ha="left",
        va="center",
        weight="bold",
    )
    ax.text(
        0.06,
        0.80,
        "The benchmark is",
        fontsize=36,
        color=INK,
        ha="left",
        va="center",
        weight="bold",
    )
    ax.text(
        0.06,
        0.72,
        "public.",
        fontsize=36,
        color=GRAPH,
        ha="left",
        va="center",
        weight="bold",
    )

    bullets = [
        "50 hand-graded questions across 5 risk categories",
        "Canonical entity-ID ground truth (no LLM judge)",
        "Reproducible end-to-end in under 90 seconds",
        "CC-BY 4.0 — drop your own retriever in and grade it",
        "Zero API spend · Kuzu + ChromaDB · CPU only",
    ]
    for i, b in enumerate(bullets):
        y = 0.60 - i * 0.055
        ax.plot([0.07], [y], marker="o", markersize=4, color=GRAPH)
        ax.text(0.10, y, b, fontsize=12.5, color=INK, ha="left", va="center")

    rect = mpatches.FancyBboxPatch(
        (0.06, 0.18),
        0.88,
        0.13,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=1.2,
        edgecolor=RULE,
        facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(
        0.5,
        0.275,
        "If you're hiring for applied AI in financial services —",
        fontsize=12,
        color=SUB,
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        0.243,
        "I'd love to walk you through the design choices.",
        fontsize=12,
        color=SUB,
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        0.205,
        "DM open  ·  Code + benchmark linked in post",
        fontsize=13,
        color=GRAPH,
        ha="center",
        va="center",
        weight="bold",
    )

    footer(ax, "5 / 5  ·  thanks for reading")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Build everything
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    results = load_results()

    builders = [
        ("slide_1_cover", lambda: slide_1_cover(results)),
        ("slide_2_table", lambda: slide_2_table(results)),
        ("slide_3_bar_chart", lambda: slide_3_bar_chart(results)),
        ("slide_4_architecture", slide_4_architecture),
        ("slide_5_cta", slide_5_cta),
    ]

    pdf_path = OUT_DIR / "entitynet_carousel.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, build in builders:
            fig = build()
            pdf.savefig(fig, facecolor=BG)
            png_path = OUT_DIR / f"{name}.png"
            fig.savefig(png_path, dpi=DPI, facecolor=BG, bbox_inches=None)
            plt.close(fig)
            print(f"  ✓ {png_path.relative_to(ROOT)}")
    print(f"  ✓ {pdf_path.relative_to(ROOT)}")
    print()
    print(f"Carousel ready: {pdf_path}")
    print("Upload that PDF to LinkedIn as a 'Document' post.")


if __name__ == "__main__":
    main()
