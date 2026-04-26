# How much did ChatGPT hurt Stack Overflow? A causal analysis.

**[Live dashboard →](https://stackoverflow-chatgpt-causal-impact.streamlit.app)** | **[Notebooks →](notebooks/)**

---

## TL;DR

Stack Overflow search interest declined approximately **16–17% relative to its counterfactual trajectory** following ChatGPT's launch in November 2022, based on two independent causal inference methods (synthetic control and OLS regression). However, a placebo-in-time test reveals that SO was already diverging from similar developer platforms before November 2022 — likely driven by the broader AI coding assistant wave beginning with GitHub Copilot (June 2022) — which means this estimate is an **upper bound**, not a precise point estimate attributable to ChatGPT alone.

---

## Why I built this

After seeing dozens of takes claiming "ChatGPT killed Stack Overflow," I wanted to know whether the data actually supported that — and more specifically, how much of the decline was causally attributable to ChatGPT versus organic trends that were already in motion. This is a causal inference problem, not just a trend analysis. I used it as an opportunity to implement synthetic control and regression-based counterfactuals on real observational data with a clean intervention event.

---

## The question

Stack Overflow's traffic had been declining for years before ChatGPT launched. The question isn't "did SO decline after ChatGPT?" (it did) — it's "how much of that decline would have happened anyway, and how much is causally attributable to ChatGPT's launch?"

That requires constructing a **counterfactual**: what would Stack Overflow's trajectory have looked like if ChatGPT had never launched?

---

## Data

**Treatment series:** Google Trends monthly search interest for `"stack overflow"`, Jan 2018 – Apr 2026, pulled via `pytrends`.

**Control series:** Google Trends for `"w3schools"` and `"geeksforgeeks"` — developer reference/tutorial sites with similar use cases to Stack Overflow, pulled in separate requests to avoid scale compression.

**Why these controls:** Both are developer-adjacent but not AI-substitutable in the same way as Q&A. They serve similar intent (finding answers to programming problems) without being direct ChatGPT targets. Pre-intervention correlations: W3Schools r = 0.71, GeeksForGeeks r = 0.64.

**GitHub excluded:** Despite a pre-period correlation of r = 0.62, GitHub's post-2022 trajectory is clearly driven by Copilot and AI-related activity — it went up 4x while SO declined, making it a contaminated control. Including it would underestimate the effect.

**Intervention date:** November 1, 2022 (ChatGPT public launch). Monthly data, so this is the first post-intervention period.

---

## Method

### Synthetic Control

Finds non-negative weights w₁, w₂ (summing to 1) over the control series that minimize pre-intervention RMSE against Stack Overflow. The optimization problem is convex (SLSQP via scipy). The resulting weighted combination — **W3Schools at 64%, GeeksForGeeks at 36%** — is the synthetic counterfactual. Pre-period RMSE: 4.76 units on a 0–100 scale.

Post-intervention causal effect = actual SO − synthetic SO.

### OLS Counterfactual

Fits `StackOverflow ~ W3Schools + GeeksForGeeks` via OLS on the pre-intervention period (pre-R² = 0.527). Projects the fitted relationship forward with 95% prediction intervals. Unlike synthetic control, OLS places no constraints on coefficients — it's a more flexible but less conservative estimate.

### Why two methods?

They make different assumptions. Synthetic control enforces convex weights (conservative, no extrapolation). OLS allows unconstrained linear combinations (more flexible, assumes linearity). If both agree, the finding is robust to modeling choice.

---

## Findings

| Method | Avg monthly effect | Relative effect |
|---|---|---|
| Synthetic Control | −6.9 units | **−16.9%** |
| OLS Counterfactual | −6.1 units | **−15.6%** |

Both methods agree within 1.3 percentage points. The effect is persistent: negative every month from January 2023 through mid-2025, deepening to about −20 units (on a 0–100 scale) by early 2025 before partially recovering in 2026.

The 2026 recovery spike is anomalous in the Google Trends data and likely reflects a short-lived event rather than genuine traffic recovery — it makes the post-period average conservative.

---

## Robustness

| Test | SC effect | OLS effect | Verdict |
|---|---|---|---|
| Baseline (Nov 2022) | −16.9% | −15.6% | — |
| Placebo-in-time (fake Nov 2021) | −25.2% | −15.1% | ⚠️ Pre-existing divergence |
| Placebo-in-unit (GitHub treated) | +143.8% | +144.2% | ✅ Effect is unit-specific |
| Alt date — Copilot GA (Jun 2022) | −19.5% | −16.4% | ⚠️ Copilot contributes |

**Placebo-in-unit** is the strongest result: applying the same method to GitHub produces a +144% effect (GitHub massively outperformed its counterfactual), which is the complete opposite of SO. Developer interest as a whole did not decline — it just shifted away from SO specifically. This validates that the SO finding isn't picking up a general trend.

**Placebo-in-time** is the most important caveat: the model detects a spurious −25% effect at a fake November 2021 intervention. This means the parallel trends assumption was weakening before ChatGPT — SO was already diverging from W3Schools and GeeksForGeeks from late 2021, likely driven by the COVID-era programming surge unwinding unevenly across platforms. The −17% estimate is therefore an **upper bound**.

**Alternative date** (Copilot GA, June 2022) produces a slightly larger SC effect (−19.5% vs −17%). The decline started before ChatGPT's public launch, consistent with a broader AI coding assistant effect rather than a single product event.

---

## Limitations

- **Google Trends is a proxy.** It tracks search interest, not actual Stack Overflow question volume, answer volume, or traffic. The signal is directionally valid but the magnitude in "search interest points" doesn't translate directly to users.
- **Only three controls.** A richer donor pool (more developer reference sites) would give the synthetic control more flexibility and likely better pre-period fit.
- **Parallel trends partially violated.** The placebo-in-time test flags a pre-existing divergence. The −17% estimate is an upper bound on the ChatGPT-attributable effect.
- **Cannot isolate ChatGPT from the broader LLM wave.** Copilot, Claude, Gemini, and others launched in the same window. The estimated effect is attributable to the AI coding assistant wave generally, with ChatGPT as the most visible event.
- **2026 anomaly.** There is an unexplained spike in SO search interest in early 2026 that likely reflects a data artifact or short-lived event, making the post-period average conservative.

---

## What I'd do with more time

- **Tag-level analysis via SEDE:** Stack Exchange Data Explorer provides actual question counts by tag (Python, JavaScript, etc.). This would let us measure whether ChatGPT-substitutable tags (debugging, syntax questions) declined faster than non-substitutable ones (architecture, algorithms).
- **Weekly granularity:** Monthly data smooths over the immediate post-launch shock. Weekly Google Trends data (pulled in shorter windows) would give a sharper estimate of when exactly the effect began.
- **Richer donor pool:** Sites like MDN, freeCodeCamp, dev.to as additional controls would improve synthetic control fit.
- **Interrupted time series as a third method:** Adding a pure ITS model (no controls needed) as an additional robustness check.

---

## Reproducing

```bash
git clone https://github.com/ritick-srivastava/stackoverflow-chatgpt-causal-impact
cd stackoverflow-chatgpt-causal-impact
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
jupyter notebook                               # run notebooks 01–04 in order
streamlit run app/streamlit_app.py             # launch dashboard
```

Google Trends data is cached in `data/raw/trends_raw.csv` after the first pull. All subsequent runs use the cache — no API calls needed.

---

## Repository structure

```
├── data/
│   ├── raw/                  # Cached Google Trends CSVs
│   └── processed/            # Model outputs, summary stats
├── notebooks/
│   ├── 01_data_collection    # pytrends pull + validation
│   ├── 02_eda                # Parallel trends, correlations
│   ├── 03_causal_impact      # SC + OLS models, headline results
│   └── 04_robustness         # Three placebo tests
├── src/
│   ├── config.py             # Dates, keywords, paths
│   ├── data.py               # Data pulling + processing
│   └── causal.py             # run_synthetic_control(), run_ols_counterfactual()
└── app/
    └── streamlit_app.py      # Interactive dashboard
```
