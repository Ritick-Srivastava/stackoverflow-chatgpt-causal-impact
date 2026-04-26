import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatGPT's Causal Impact on Stack Overflow",
    page_icon="📉",
    layout="wide",
)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA = Path(__file__).parent.parent / "data" / "processed"


@st.cache_data
def load_data():
    ols = pd.read_csv(DATA / "ols_results.csv", index_col=0, parse_dates=True)
    sc  = pd.read_csv(DATA / "sc_results.csv",  index_col=0, parse_dates=True)
    with open(DATA / "summary.json")     as f: summary    = json.load(f)
    with open(DATA / "robustness.json")  as f: robustness = json.load(f)
    return ols, sc, summary, robustness


ols, sc, summary, robustness = load_data()
intervention = pd.Timestamp(summary["intervention_date"])

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📉 How much did ChatGPT hurt Stack Overflow?")
st.markdown(
    "A causal inference analysis using **synthetic control** and **OLS regression**. "
    "Two independent methods isolate the intervention effect from organic decline."
)
st.divider()

# ── Headline metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Synthetic Control estimate",
    f"{summary['sc_rel_effect_pct']:.1f}%",
    help="Avg monthly SO search interest relative to synthetic counterfactual, post-Nov 2022",
)
c2.metric(
    "OLS estimate",
    f"{summary['ols_rel_effect_pct']:.1f}%",
    help="Avg monthly SO search interest relative to OLS projected counterfactual, post-Nov 2022",
)
c3.metric(
    "SC donor weights",
    f"W3S {summary['sc_weights']['W3Schools']:.0%} / GfG {summary['sc_weights']['GeeksForGeeks']:.0%}",
    help="Optimal weights assigned to each control series",
)
c4.metric(
    "Pre-period RMSE (SC)",
    f"{summary['sc_pre_rmse']:.2f}",
    help="Pre-intervention fit quality — lower is better",
)

st.divider()

# ── Controls ───────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([3, 1])

with col_r:
    st.markdown("**Chart options**")
    method      = st.radio("Method", ["Synthetic Control", "OLS Counterfactual"], index=0)
    show_ci     = st.checkbox("Show prediction interval", value=True)
    post_only   = st.checkbox("Zoom to post-intervention", value=False)

# ── Main chart ─────────────────────────────────────────────────────────────────
with col_l:
    if method == "Synthetic Control":
        actual        = sc["actual"]
        counterfactual = sc["synthetic"]
        ci_lower      = None
        ci_upper      = None
        cf_label      = "Synthetic counterfactual"
        cf_color      = "#1f77b4"
    else:
        actual         = ols["actual"]
        counterfactual = ols["counterfactual"]
        ci_lower       = ols["ci_lower"]
        ci_upper       = ols["ci_upper"]
        cf_label       = "OLS counterfactual"
        cf_color       = "#1f77b4"

    if post_only:
        mask = actual.index >= intervention
        actual         = actual[mask]
        counterfactual = counterfactual[mask]
        if ci_lower is not None:
            ci_lower = ci_lower[mask]
            ci_upper = ci_upper[mask]

    fig = go.Figure()

    if show_ci and ci_lower is not None:
        fig.add_trace(go.Scatter(
            x=list(counterfactual.index) + list(counterfactual.index[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill="toself", fillcolor="rgba(31,119,180,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% prediction interval", showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=counterfactual.index, y=counterfactual,
        line=dict(color=cf_color, dash="dash", width=2),
        name=cf_label,
    ))
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual,
        line=dict(color="#F48024", width=2.5),
        name="Actual (Stack Overflow)",
    ))

    if not post_only:
        fig.add_vline(
            x=intervention.timestamp() * 1000,
            line_dash="dash", line_color="red", line_width=1.5,
            annotation_text="ChatGPT launch (Nov 2022)",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{method}: Stack Overflow actual vs counterfactual",
        xaxis_title="Date",
        yaxis_title="Search interest (0–100, Google Trends)",
        height=460,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=80),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Effect over time ───────────────────────────────────────────────────────────
st.markdown("#### Monthly causal effect (actual − counterfactual)")

if method == "Synthetic Control":
    effect_series = sc["effect"]
else:
    effect_series = ols["effect"]

post_effect = effect_series[effect_series.index >= intervention]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=post_effect.index, y=post_effect,
    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in post_effect],
    name="Monthly effect",
))
fig2.add_hline(y=0, line_color="black", line_width=1)
fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Effect (search interest units)",
    height=300,
    template="plotly_white",
    showlegend=False,
    margin=dict(t=20),
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Robustness ─────────────────────────────────────────────────────────────────
st.markdown("#### Robustness checks")

rob_df = pd.DataFrame({
    "Test": [
        "Baseline (Nov 2022, SO treated)",
        "Placebo-in-time (fake Nov 2021)",
        "Placebo-in-unit (GitHub treated)",
        "Alt date — Copilot GA (Jun 2022)",
    ],
    "SC effect": [
        f"{summary['sc_rel_effect_pct']:+.1f}%",
        f"{robustness['placebo_time_sc']:+.1f}%",
        f"{robustness['placebo_unit_sc']:+.1f}%",
        f"{robustness['copilot_cut_sc']:+.1f}%",
    ],
    "OLS effect": [
        f"{summary['ols_rel_effect_pct']:+.1f}%",
        f"{robustness['placebo_time_ols']:+.1f}%",
        f"{robustness['placebo_unit_ols']:+.1f}%",
        f"{robustness['copilot_cut_ols']:+.1f}%",
    ],
    "Verdict": ["—", "⚠️ Pre-existing divergence", "✅ Opposite sign", "⚠️ Copilot contributes"],
})
st.dataframe(rob_df, use_container_width=True, hide_index=True)

st.divider()

# ── Methodology ────────────────────────────────────────────────────────────────
with st.expander("Methodology"):
    st.markdown("""
**Data:** Google Trends monthly search interest for "stack overflow" (treatment),
"w3schools", and "geeksforgeeks" (controls). Jan 2018 – Apr 2026.
GitHub excluded from causal model — Copilot-driven growth contaminates it post-2022.

**Synthetic Control:** Finds non-negative weights (summing to 1) over control series that
minimize pre-intervention RMSE against Stack Overflow. Projects that weighted combination
forward as the counterfactual. Optimal weights: W3Schools 64%, GeeksForGeeks 36%.

**OLS Counterfactual:** Fits `StackOverflow ~ W3Schools + GeeksForGeeks` on the
pre-intervention period (pre-R² = 0.527), then projects forward with 95% prediction intervals.

**Intervention date:** November 1, 2022 (ChatGPT public launch).

**Causal identification assumption:** In the absence of ChatGPT, Stack Overflow's search
interest would have continued to track the weighted combination of controls.
This assumption is partially violated — the placebo-in-time test detects a pre-existing
divergence beginning ~2021, suggesting the -16–17% estimate is an upper bound.
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "Data: Google Trends via pytrends · Methods: synthetic control + OLS · "
    "By Ritick Srivastava · "
    "[GitHub](https://github.com/ritick-srivastava/stackoverflow-chatgpt-causal-impact)"
)
