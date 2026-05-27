"""EntityNet dashboard — experimental prototype.

This Streamlit app is not the headline artifact of the project — the
50-question benchmark is. It lives here as an exploratory UI for local
inspection of the graph and side-by-side retriever comparisons, and is
tracked as a roadmap item rather than a primary deliverable.

Pages:
1. Overview      — KPI cards, graph stats, top high-risk entities.
2. Risk search   — entity lookup → risk report + interactive subgraph.
3. RAG compare   — side-by-side Vector / Graph / Hybrid on one question.
4. Benchmark     — F1 by retriever and by category.
5. Graph explore — full entity table + 2-hop neighbourhood.
6. Methodology   — what was built and why.

Install the optional extras first:  uv pip install -e ".[dashboard]"
Then run with:                      streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import plotly.express as px
import streamlit as st
from styles import ENTITY_COLORS, PALETTE, hero, inject_css, kpi_card, risk_badge, section

from entitynet.config import BENCHMARK_RESULTS_PATH
from entitynet.db import get_conn
from entitynet.graph.builder import graph_stats
from entitynet.graph.queries import (
    find_entity_by_name,
    k_hop_neighborhood,
    subgraph_nodes_and_edges,
)
from entitynet.retrieve.graph_rag import graph_retrieve
from entitynet.retrieve.hybrid import hybrid_retrieve
from entitynet.retrieve.risk_report import risk_report
from entitynet.retrieve.vector import vector_retrieve

st.set_page_config(
    page_title="EntityNet · GraphRAG for Entity Risk",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# =========================================================================
# Cached helpers
# =========================================================================


@st.cache_data(ttl=300)
def cached_graph_stats() -> dict[str, int]:
    try:
        return graph_stats()
    except Exception:
        return {"person": 0, "company": 0, "sanction": 0, "newsarticle": 0}


@st.cache_data(ttl=300)
def cached_entity_table() -> pd.DataFrame:
    rows = []
    try:
        with get_conn() as conn:
            for kind, prefix in [("Person", "P"), ("Company", "C")]:
                res = conn.execute(f"MATCH (n:{kind}) RETURN n.id AS nid, n.name AS nname")
                while res.has_next():
                    r = res.get_next()
                    rows.append({"id": str(r[0]), "label": kind, "name": str(r[1]), "prefix": prefix})
    except Exception:
        return pd.DataFrame(columns=["id", "label", "name", "prefix"])
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


@st.cache_data(ttl=600)
def cached_risk_report(entity_id: str):
    return risk_report(entity_id)


@st.cache_data(ttl=600)
def cached_top_risk(limit: int = 8) -> pd.DataFrame:
    """Compute a risk report for every entity and return the top N."""
    df = cached_entity_table()
    rows = []
    for eid in df["id"].tolist():
        try:
            r = cached_risk_report(eid)
            rows.append(
                {
                    "id": r.entity_id,
                    "name": r.entity_name,
                    "level": r.risk_level.value,
                    "score": r.risk_score,
                    "direct": len(r.direct_sanctions),
                    "chains": len(r.indirect_sanctions_paths),
                    "media": len(r.adverse_media),
                }
            )
        except Exception:
            continue
    out = pd.DataFrame(rows).sort_values("score", ascending=False).head(limit).reset_index(drop=True)
    return out


def _entity_label(entity_id: str) -> str:
    if not entity_id:
        return ""
    df = cached_entity_table()
    hit = df[df["id"] == entity_id]
    if hit.empty:
        return entity_id
    return f"{entity_id} — {hit.iloc[0]['name']}"


# =========================================================================
# Subgraph viz
# =========================================================================


def render_subgraph(focal_id: str, hops: int = 2, height: int = 520) -> None:
    """Render a coloured pyvis subgraph for `focal_id`."""
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("pyvis is not installed.")
        return
    from streamlit.components.v1 import html

    nodes, edges = subgraph_nodes_and_edges(focal_id, hops=hops)
    if not nodes:
        st.info("No subgraph found for this entity.")
        return

    net = Network(height=f"{height}px", width="100%", bgcolor="#FFFFFF", font_color="#0F172A", directed=True)
    net.barnes_hut(gravity=-2400, central_gravity=0.25, spring_length=160, spring_strength=0.04)
    net.set_options(
        """
        {
          "interaction": {"hover": true, "tooltipDelay": 80},
          "physics": {"stabilization": {"iterations": 180}},
          "edges": {"smooth": {"type": "dynamic"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}}
        }
        """
    )

    for n in nodes:
        color = ENTITY_COLORS.get(n["label"], "#94A3B8")
        size = 30 if n["id"] == focal_id else 20
        border = "#1E293B" if n["id"] == focal_id else color
        net.add_node(
            n["id"],
            label=f"{n['id']}\n{n['name'][:22]}",
            title=f"{n['label']}: {n['name']}",
            color={"background": color, "border": border, "highlight": {"background": color, "border": "#0F172A"}},
            borderWidth=3 if n["id"] == focal_id else 1,
            size=size,
            font={"size": 12, "color": "#0F172A"},
        )

    edge_colors = {
        "OWNS": "#6366F1",
        "DIRECTOR_OF": "#10B981",
        "BUSINESS_PARTNER_OF": "#F59E0B",
        "RELATIVE_OF": "#EC4899",
        "SANCTIONS_TARGET": "#EF4444",
        "MENTIONED_IN": "#94A3B8",
    }
    for src, dst, lbl in edges:
        net.add_edge(src, dst, label=lbl, title=lbl, color=edge_colors.get(lbl, "#94A3B8"), width=2)

    html(net.generate_html(notebook=False), height=height + 10, scrolling=False)


# =========================================================================
# Sidebar
# =========================================================================


with st.sidebar:
    st.markdown(
        """
        <div style="padding: 6px 4px 14px 4px;">
            <div style="font-size:22px; font-weight:800; letter-spacing:-0.02em;">🕸️ EntityNet</div>
            <div style="font-size:12px; opacity:0.7; margin-top:2px;">GraphRAG · Adverse Media · Entity Risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        (
            "Overview",
            "Risk search",
            "RAG comparison",
            "Benchmark",
            "Graph explorer",
            "Methodology",
        ),
        label_visibility="collapsed",
    )

    st.markdown("---")
    stats = cached_graph_stats()
    total_nodes = sum(stats.values())
    if total_nodes:
        st.markdown(
            f"""
            <div style="font-size:11px; opacity:0.6; text-transform:uppercase; letter-spacing:0.08em;">Graph stats</div>
            <div style="font-size:13px; margin-top:6px; opacity:0.9;">
              👤 Persons · <b>{stats.get('person', 0)}</b><br>
              🏢 Companies · <b>{stats.get('company', 0)}</b><br>
              ⚠️ Sanctions · <b>{stats.get('sanction', 0)}</b><br>
              📰 News · <b>{stats.get('newsarticle', 0)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Graph empty. Run `make build`.")


# =========================================================================
# Page 1: Overview
# =========================================================================


if page == "Overview":
    hero(
        "GraphRAG for Adverse Media &amp; Entity Risk",
        "A knowledge-graph RAG system that surfaces hidden ownership chains, sanctioned-relative links, "
        "and adverse media — questions flat vector RAG fundamentally cannot answer.",
        chips=["Kuzu graph", "ChromaDB vector", "spaCy NER", "50-question benchmark", "$0 to run"],
    )

    stats = cached_graph_stats()
    cols = st.columns(4)
    cards = [
        ("Persons", stats.get("person", 0), "individuals in the graph"),
        ("Companies", stats.get("company", 0), "incl. shells & holdings"),
        ("Sanctions", stats.get("sanction", 0), "OFAC, EU, UK, UN"),
        ("News articles", stats.get("newsarticle", 0), "with linked entities"),
    ]
    for col, (label, val, sub) in zip(cols, cards, strict=False):
        col.markdown(kpi_card(label, str(val), sub), unsafe_allow_html=True)

    # Top risk entities
    section("Top risk entities", "ranked by structured risk score")
    top = cached_top_risk(limit=10)
    if top.empty:
        st.info("No risk data yet. Build the graph first.")
    else:
        # Render a card grid
        for _, row in top.iterrows():
            cols = st.columns([1.5, 2.6, 1, 1, 1, 1])
            cols[0].markdown(f"<b>{row['id']}</b>", unsafe_allow_html=True)
            cols[1].markdown(row["name"])
            cols[2].markdown(risk_badge(row["level"]), unsafe_allow_html=True)
            cols[3].markdown(f"<b>{row['score']}</b><span style='color:#94A3B8'> /100</span>", unsafe_allow_html=True)
            chip_style = "background:#F1F5F9;color:#475569;padding:2px 8px;border-radius:999px;font-size:12px;"
            cols[4].markdown(
                f"<span style='{chip_style}'>{row['chains']} chains · {row['direct']} direct</span>",
                unsafe_allow_html=True,
            )
            cols[5].markdown(
                f"<span style='{chip_style}'>{row['media']} news</span>",
                unsafe_allow_html=True,
            )

    # Benchmark snapshot if available
    if BENCHMARK_RESULTS_PATH.exists():
        section("Benchmark snapshot", "open the Benchmark page for the full chart")
        summary = json.loads(BENCHMARK_RESULTS_PATH.read_text())
        rows = []
        for name, vals in summary["overall"].items():
            rows.append({"Retriever": name, "F1": vals["f1_mean"], "Recall": vals["recall_mean"]})
        snap = pd.DataFrame(rows).set_index("Retriever")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='en-card'><h3>Overall F1</h3>", unsafe_allow_html=True)
            st.dataframe(
                snap,
                use_container_width=True,
                column_config={
                    "F1": st.column_config.ProgressColumn("F1", min_value=0.0, max_value=1.0, format="%.3f"),
                    "Recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                },
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            fig = px.bar(
                snap.reset_index(),
                x="Retriever",
                y="F1",
                color="Retriever",
                color_discrete_sequence=["#94A3B8", "#6366F1", "#10B981"],
                text_auto=".2f",
            )
            fig.update_layout(
                height=280,
                margin={"t": 10, "l": 10, "r": 10, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis={"range": [0, 1], "title": ""},
                xaxis={"title": ""},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================================
# Page 2: Risk search
# =========================================================================


elif page == "Risk search":
    hero(
        "Entity risk search",
        "Type a company or person name. EntityNet traverses the knowledge graph for direct sanctions, "
        "indirect ownership chains, shared-director risk, family links, partner exposure, and adverse media.",
        chips=["Multi-hop traversal", "Structured risk score", "Subgraph visualization"],
    )

    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Entity name", value="Delta Energy Solutions", label_visibility="collapsed", placeholder="e.g. Delta Energy Solutions")
    hops = c2.selectbox("Subgraph hops", options=[1, 2, 3], index=1)

    if q:
        hits = find_entity_by_name(q)
        if not hits:
            st.warning("No entity matched that name.")
            st.stop()
        chosen_id = st.selectbox(
            "Matched entities",
            options=[h[0] for h in hits],
            format_func=lambda eid: next((f"{h[0]} · {h[1]} · {h[2]}" for h in hits if h[0] == eid), eid),
        )

        report = cached_risk_report(chosen_id)

        # KPI strip
        cols = st.columns(4)
        cols[0].markdown(
            kpi_card("Risk score", f"{report.risk_score}", "out of 100"),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f"""<div class="en-card">
                <h3>Risk level</h3>
                <div style="margin-top:8px">{risk_badge(report.risk_level.value)}</div>
                <div class="en-sub">heuristic, weighted by graph traversals</div>
            </div>""",
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            kpi_card("Direct sanctions", str(len(report.direct_sanctions)), "OFAC / EU / UK / UN / Entity List"),
            unsafe_allow_html=True,
        )
        cols[3].markdown(
            kpi_card("Indirect chains", str(len(report.indirect_sanctions_paths)), "≤3-hop ownership paths"),
            unsafe_allow_html=True,
        )

        st.markdown(f"<div class='en-card' style='margin-top:14px'><b>Summary.</b> {report.summary}</div>", unsafe_allow_html=True)

        section("Risk graph", f"{hops}-hop neighbourhood")
        render_subgraph(chosen_id, hops=hops)

        # Legend
        legend = " &nbsp;·&nbsp; ".join(
            f"<span style='display:inline-flex;align-items:center;gap:6px;font-size:12px;'>"
            f"<span style='width:10px;height:10px;border-radius:999px;background:{c};display:inline-block'></span>{lbl}</span>"
            for lbl, c in [
                ("Person", PALETTE["person"]),
                ("Company", PALETTE["company"]),
                ("Sanction", PALETTE["sanction"]),
                ("News", PALETTE["news"]),
                ("Focal", PALETTE["focal"]),
            ]
        )
        st.markdown(f"<div style='color:#475569;margin-top:-6px'>{legend}</div>", unsafe_allow_html=True)

        # Tabs for details
        tab_chains, tab_media, tab_partners = st.tabs(
            ["Ownership chains", "Adverse media", "High-risk connections"]
        )
        with tab_chains:
            if report.indirect_sanctions_paths:
                for path in report.indirect_sanctions_paths:
                    arrow_path = " &nbsp;→&nbsp; ".join(f"<b>{p}</b>" for p in path)
                    st.markdown(f"<div class='en-path'>{arrow_path}</div>", unsafe_allow_html=True)
                    st.caption(" → ".join(_entity_label(p) for p in path))
            else:
                st.write("No indirect chains.")

        with tab_media:
            if not report.adverse_media:
                st.write("No adverse-media mentions.")
            for nid in report.adverse_media:
                with get_conn() as conn:
                    res = conn.execute(
                        "MATCH (a:NewsArticle) WHERE a.id = $aid "
                        "RETURN a.title AS t, a.source AS s, a.published_date AS d, a.body AS b",
                        {"aid": nid},
                    )
                    if res.has_next():
                        r = res.get_next()
                        st.markdown(
                            f"<div class='en-card' style='margin-bottom:8px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                            f"<b>{nid} · {r[0]}</b>"
                            f"<span style='color:#94A3B8;font-size:12px'>{r[1]} · {r[2]}</span>"
                            f"</div>"
                            f"<div style='color:#475569;margin-top:8px;font-size:13.5px'>{str(r[3])[:520]}{'…' if len(str(r[3])) > 520 else ''}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        with tab_partners:
            if report.high_risk_connections:
                for line in report.high_risk_connections:
                    st.markdown(f"<div class='en-path'>{line}</div>", unsafe_allow_html=True)
            else:
                st.write("No high-risk personal/partner connections.")


# =========================================================================
# Page 3: RAG comparison
# =========================================================================


elif page == "RAG comparison":
    hero(
        "Vector RAG · GraphRAG · Hybrid",
        "Run the same multi-hop risk question through three retrievers. Where graph traversal helps, "
        "it shows up here as longer evidence paths and tighter entity sets.",
        chips=["Side-by-side", "Latency tracked", "Entity-set output"],
    )

    presets = [
        "Who is the ultimate beneficial owner of Delta Energy Solutions?",
        "Does Falcon Industries have any indirect sanctions exposure?",
        "What companies share a director with Iron Mountain Holdings?",
        "Does Phoenix Resources have indirect sanctions exposure through partners?",
        "Through what family connection is Robert Chen Ventures linked to a US-listed entity?",
    ]
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Preset question", options=presets + ["(custom)"], label_visibility="collapsed")
    run_btn = c2.button("Run all retrievers", type="primary", use_container_width=True)

    if choice == "(custom)":
        question = st.text_input("Your question", value="")
    else:
        question = choice
        st.markdown(f"<div class='en-card en-card-soft'>{question}</div>", unsafe_allow_html=True)

    if run_btn and question:
        a, b, c = st.columns(3)

        try:
            vr = vector_retrieve(question)
            vec_items = "".join(f"<li>{_entity_label(e)}</li>" for e in vr.entity_ids[:8]) or "<li>(none)</li>"
            a.markdown(
                f"""<div class="en-compare">
                    <h4 style="color:#475569">Vector RAG</h4>
                    <div class="en-meta">{vr.latency_ms} ms · top-k embeddings</div>
                    <ul>{vec_items}</ul>
                </div>""",
                unsafe_allow_html=True,
            )
        except Exception as e:
            a.error(str(e))

        gr = graph_retrieve(question)
        graph_items = "".join(f"<li>{_entity_label(e)}</li>" for e in gr.entity_ids[:8]) or "<li>(none)</li>"
        paths_html = ""
        if gr.paths:
            paths_html = "<div style='margin-top:10px;font-size:12px;color:#475569'><b>Paths</b></div>"
            for p in gr.paths[:5]:
                arrow = " → ".join(p)
                paths_html += f"<div class='en-path' style='margin-top:4px'>{arrow}</div>"
        b.markdown(
            f"""<div class="en-compare" style="border:1.5px solid #6366F1;box-shadow:0 8px 24px rgba(99,102,241,0.15)">
                <h4 style="color:#6366F1">GraphRAG ★</h4>
                <div class="en-meta">{gr.latency_ms} ms · multi-hop traversal</div>
                <ul>{graph_items}</ul>
                {paths_html}
            </div>""",
            unsafe_allow_html=True,
        )

        hr = hybrid_retrieve(question)
        hyb_items = "".join(f"<li>{_entity_label(e)}</li>" for e in hr.entity_ids[:8]) or "<li>(none)</li>"
        c.markdown(
            f"""<div class="en-compare">
                <h4 style="color:#10B981">Hybrid</h4>
                <div class="en-meta">{hr.latency_ms} ms · rank fusion</div>
                <ul>{hyb_items}</ul>
            </div>""",
            unsafe_allow_html=True,
        )


# =========================================================================
# Page 4: Benchmark
# =========================================================================


elif page == "Benchmark":
    hero(
        "Benchmark · 50 questions · entity-F1 graded",
        "Three retrievers, one ground-truth set per question. GraphRAG roughly doubles vector-RAG F1, "
        "with the largest gap on multi-hop categories.",
        chips=["Deterministic grading", "No LLM judge", "Reproducible in ≤90s"],
    )

    if not BENCHMARK_RESULTS_PATH.exists():
        st.info("Run `make benchmark` first.")
        st.stop()

    summary = json.loads(BENCHMARK_RESULTS_PATH.read_text())

    # Overall KPIs
    cols = st.columns(3)
    for col, name in zip(cols, ("vector", "graph", "hybrid"), strict=False):
        if name in summary["overall"]:
            v = summary["overall"][name]
            tag = {
                "vector": ("Vector RAG", "#94A3B8", "baseline"),
                "graph": ("GraphRAG", "#6366F1", "★ headline retriever"),
                "hybrid": ("Hybrid", "#10B981", "rank fusion"),
            }[name]
            col.markdown(
                f"""<div class="en-card">
                    <h3 style="color:{tag[1]}">{tag[0]}</h3>
                    <div class="en-value">{v['f1_mean']:.3f}</div>
                    <div class="en-sub">F1 · recall {v['recall_mean']:.2f} · p50 {v['latency_ms_p50']:.0f}ms · {tag[2]}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    section("F1 by category", "the chart for LinkedIn post #1")
    rows = []
    for cat, perret in summary["by_category"].items():
        for ret, vals in perret.items():
            rows.append({"Category": cat, "Retriever": ret, "F1": vals["f1_mean"], "N": vals["n"]})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Category",
        y="F1",
        color="Retriever",
        barmode="group",
        text_auto=".2f",
        color_discrete_map={"vector": "#94A3B8", "graph": "#6366F1", "hybrid": "#10B981"},
        category_orders={
            "Category": [
                "direct_sanctions",
                "ownership_chain",
                "hidden_connection",
                "risk_propagation",
                "adverse_media",
            ]
        },
    )
    fig.update_layout(
        height=420,
        margin={"t": 20, "l": 10, "r": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"range": [0, 1], "title": "F1 score"},
        xaxis={"title": ""},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Per-question results", "sortable; click any column header")
    pq = pd.DataFrame(summary["per_question"])
    st.dataframe(
        pq,
        use_container_width=True,
        hide_index=True,
        column_config={
            "f1": st.column_config.ProgressColumn("F1", min_value=0.0, max_value=1.0, format="%.3f"),
            "precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
            "recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )


# =========================================================================
# Page 5: Graph explorer
# =========================================================================


elif page == "Graph explorer":
    hero(
        "Graph explorer",
        "Browse every entity in the knowledge graph. Pick any row to inspect its 2-hop neighbourhood.",
        chips=["Live Kuzu queries"],
    )

    df = cached_entity_table()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)
    with c2:
        selected = st.selectbox("Inspect", df["id"].tolist(), format_func=_entity_label)
        if selected:
            n = len(k_hop_neighborhood(selected, hops=2))
            st.markdown(
                f"<div class='en-card'><h3>{selected}</h3>"
                f"<div class='en-value' style='font-size:22px'>{n}</div>"
                f"<div class='en-sub'>nodes within 2 hops</div></div>",
                unsafe_allow_html=True,
            )

    if selected:
        section("2-hop neighbourhood", f"around {_entity_label(selected)}")
        render_subgraph(selected, hops=2, height=560)


# =========================================================================
# Page 6: Methodology
# =========================================================================


else:
    hero(
        "Methodology",
        "What was built and why — design choices, limitations, and the natural follow-ons.",
    )

    st.markdown(
        """
<div class="en-card">
<h3 style="color:#475569">Why this project exists</h3>
<p>99% of "advanced RAG" demos in 2026 are still <b>flat vector + reranker</b>. GraphRAG is the genuinely
cutting-edge alternative — and it maps directly to AML / KYC / adverse-media work at banks, fintechs,
payments networks, and regtech.</p>
</div>

<div class="en-card" style="margin-top:14px">
<h3 style="color:#475569">How the comparison works</h3>
<p>Every question has a ground-truth set of canonical entity IDs. Every retriever returns a set of entity
IDs. We score <b>F1</b> — the same metric for every retriever, no LLM judge involved.
Fully reproducible.</p>
</div>

<div class="en-card" style="margin-top:14px">
<h3 style="color:#475569">Retrievers</h3>
<table style="width:100%; border-collapse:collapse; font-size:14px;">
<thead><tr style="text-align:left; color:#94A3B8; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">
<th style="padding:8px 0;">Retriever</th><th>Stack</th><th>Good at</th></tr></thead>
<tbody>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;"><b>Vector RAG</b></td>
<td>sentence-transformers + ChromaDB</td><td>Surface-form similarity</td></tr>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;"><b>GraphRAG ★</b></td>
<td>Kuzu + Cypher traversal patterns</td><td>Multi-hop ownership, hidden links</td></tr>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;"><b>Hybrid</b></td>
<td>Union + overlap-bonus reranking</td><td>Best of both</td></tr>
</tbody></table>
</div>

<div class="en-card" style="margin-top:14px">
<h3 style="color:#475569">Deliberately not in v1</h3>
<table style="width:100%; border-collapse:collapse; font-size:14px;">
<thead><tr style="text-align:left; color:#94A3B8; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">
<th style="padding:8px 0;">Skipped</th><th>Why</th></tr></thead>
<tbody>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;">Real OpenSanctions ingestion</td>
<td>Synthetic data lets the demo cold-start in 30s.</td></tr>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;">LLM-based entity extraction</td>
<td>spaCy + fuzzy linking is more transparent and reproducible.</td></tr>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;">Neo4j</td>
<td>Kuzu is embedded — no server, no Docker, no friction.</td></tr>
<tr style="border-top:1px solid #E2E8F0;"><td style="padding:10px 0;">Reranker on vector RAG</td>
<td>The point is to show what flat vector RAG cannot do.</td></tr>
</tbody></table>
</div>

<div class="en-card" style="margin-top:14px">
<h3 style="color:#475569">Honest limitations</h3>
<ul>
<li>Synthetic data — absolute numbers will shift on a real corpus.</li>
<li>Intent detection is keyword-based — an LLM router would handle phrasing edge cases better.</li>
<li>Entity linker is name-string-based — production systems also use country, DOB, identifiers.</li>
<li>Benchmark covers only English-language risk questions.</li>
</ul>
</div>
        """,
        unsafe_allow_html=True,
    )
