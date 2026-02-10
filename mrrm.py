"""
Macro Risk Regime Monitor (MRRM)
================================
Real-time macro-financial stress dashboard combining PCA-based composite
stress indexing with statistical regime detection (HMM/GMM).

Methodology
-----------
1. Four stress factors are computed from market data:
   - Implied volatility (VIX)
   - Realized volatility (S&P 500, annualized)
   - Credit spread momentum (HYG/IEF ratio, inverted)
   - Yield curve delta (10Y-3M slope rate of change, inverted)

2. Each factor is z-scored on a rolling window, then aggregated via PCA.
   The first principal component serves as the composite stress index.
   Sign convention enforces positive = stress (correlation with VIX).

3. The stress index is EMA-smoothed, then classified into regimes
   using Hidden Markov Models (preferred) or Gaussian Mixture Models.

References
----------
- OFR Financial Stress Index methodology (dynamic PCA weighting)
- Kritzman, Page & Turkington (2012), "Regime Shifts: Implications for
  Dynamic Strategies", Financial Analysts Journal

Requirements
------------
    pip install streamlit pandas numpy yfinance plotly scikit-learn hmmlearn

Usage
-----
    streamlit run mrrm.py
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "UST10Y": "^TNX",
    "UST3M": "^IRX",
    "HYG": "HYG",
    "IEF": "IEF",
}

_REQUIRED = ("SPX", "VIX")

REGIME_PALETTE = {
    "RISK_ON": "#00C853",
    "NEUTRAL": "#FFD600",
    "STRESS": "#FF1744",
}

REFERENCE_EVENTS = {
    "2008-10-10": "GFC — Lehman aftermath",
    "2011-08-08": "US debt ceiling crisis",
    "2015-08-24": "CNY devaluation shock",
    "2018-02-06": "Volmageddon",
    "2020-03-16": "COVID-19 crash",
    "2022-06-13": "Fed rate hiking cycle",
    "2024-08-05": "JPY carry unwind",
}


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def _extract_close(frame: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract close prices handling yfinance MultiIndex variations."""
    if frame is None or frame.empty:
        raise ValueError("Empty dataframe")

    if isinstance(frame.columns, pd.MultiIndex):
        for field in ("Close", "Adj Close"):
            try:
                if (field, ticker) in frame.columns:
                    s = frame[(field, ticker)].dropna()
                    if not s.empty:
                        return s
            except (KeyError, TypeError):
                continue

        flat = frame.copy()
        flat.columns = [
            "_".join(str(c) for c in col).strip("_") for col in frame.columns
        ]
        for candidate in (f"Close_{ticker}", f"Adj Close_{ticker}"):
            if candidate in flat.columns:
                s = flat[candidate].dropna()
                if not s.empty:
                    return s

        raise ValueError(f"Cannot resolve Close for {ticker} in MultiIndex")

    for field in ("Close", "Adj Close"):
        if field in frame.columns:
            return frame[field].dropna()

    raise ValueError("No Close column found")


def _rescale_yield(series: pd.Series) -> pd.Series:
    """Correct basis-point scaling when yields are reported x10."""
    if np.nanmedian(series) > 15:
        return series / 10.0
    return series


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_market_data(start_date: str) -> tuple[pd.DataFrame, list[str]]:
    """Download and align market data from Yahoo Finance."""
    series_map, errors = {}, []

    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=start_date, progress=False, threads=False)
            s = _extract_close(raw, ticker)
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                raise ValueError("No valid data points")
            s.name = name
            series_map[name] = s
        except Exception as exc:
            errors.append(f"{name} ({ticker}): {exc}")

    if not series_map:
        raise RuntimeError("All downloads failed:\n" + "\n".join(errors))

    combined = pd.concat(series_map.values(), axis=1).sort_index().ffill()
    missing = [t for t in _REQUIRED if t not in combined.columns]
    if missing:
        raise RuntimeError(f"Missing required series: {missing}")

    return combined.dropna(subset=list(_REQUIRED)), errors


# ---------------------------------------------------------------------------
# Stress indicators
# ---------------------------------------------------------------------------

def build_stress_indicators(
    df: pd.DataFrame,
    momentum_window: int,
    vol_window: int,
) -> pd.DataFrame:
    """
    Compute four stress factors, each oriented so that higher = more stress.

    - VIX: level (implied vol)
    - Credit: inverted HY/Treasury relative performance
    - dCurve: rate of yield curve flattening/inversion
    - RVol: realized equity volatility (annualized)
    """
    cols = df.columns
    indicators = {}

    indicators["VIX"] = df["VIX"]

    if "HYG" in cols and "IEF" in cols:
        hy_ratio = df["HYG"] / df["IEF"]
        indicators["Credit"] = -hy_ratio.pct_change(momentum_window) * 100

    if "UST10Y" in cols and "UST3M" in cols:
        slope = _rescale_yield(df["UST10Y"]) - _rescale_yield(df["UST3M"])
        indicators["dCurve"] = -slope.diff(momentum_window)

    indicators["RVol"] = (
        df["SPX"].pct_change(1).rolling(vol_window).std() * np.sqrt(252) * 100
    )

    return pd.DataFrame(indicators, index=df.index)


# ---------------------------------------------------------------------------
# PCA stress index
# ---------------------------------------------------------------------------

def compute_stress_index(
    indicators: pd.DataFrame,
    lookback: int,
    clip_z: float,
    ema_span: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, np.ndarray]:
    """
    Build rolling z-scores, extract PC1 as composite stress index,
    enforce sign convention, and apply EMA smoothing.

    Returns (smoothed, raw, z_scores, loadings, explained_variance_ratios).
    """
    z_scores = pd.DataFrame(index=indicators.index)
    for col in indicators.columns:
        mu = indicators[col].rolling(lookback).mean()
        sigma = indicators[col].rolling(lookback).std()
        z_scores[col] = ((indicators[col] - mu) / sigma).clip(-clip_z, clip_z)

    valid = z_scores.dropna()
    if len(valid) < 100:
        raise ValueError(f"Only {len(valid)} valid observations — need >= 100")

    pca = PCA(n_components=valid.shape[1])
    scores = pca.fit_transform(valid.values)
    raw = pd.Series(scores[:, 0], index=valid.index)

    if "VIX" in valid.columns:
        rho = raw.corr(valid["VIX"])
        if not np.isnan(rho) and rho < 0:
            raw = -raw
            pca.components_[0] *= -1

    smoothed = raw.ewm(span=ema_span, min_periods=1).mean()
    loadings = pd.Series(pca.components_[0], index=valid.columns)

    return smoothed, raw, valid, loadings, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _build_label_map(ordered_states: list, n_states: int) -> dict:
    """Map state indices to semantic labels ordered by stress level."""
    if n_states == 2:
        return {ordered_states[0]: "RISK_ON", ordered_states[1]: "STRESS"}
    mapping = {ordered_states[0]: "RISK_ON", ordered_states[-1]: "STRESS"}
    for s in ordered_states[1:-1]:
        mapping[s] = "NEUTRAL"
    return mapping


def _smooth_regimes(regime: pd.Series, min_duration: int) -> pd.Series:
    """Eliminate spurious regime flickers shorter than min_duration days."""
    values = regime.values.copy()
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j < n and values[j] == values[i]:
            j += 1
        if (j - i) < min_duration and i > 0:
            values[i:j] = values[i - 1]
        i = j
    return pd.Series(values, index=regime.index)


def detect_regimes(
    stress: pd.Series,
    n_states: int,
    method: str,
    min_duration: int,
) -> tuple[pd.Series, pd.DataFrame | None, str]:
    """
    Classify stress index into regimes via HMM, GMM, or quantile fallback.

    Returns (regime_series, probabilities_df_or_None, method_tag).
    """
    X = stress.dropna().values.reshape(-1, 1)
    idx = stress.dropna().index

    if method == "HMM" and _HMM_AVAILABLE:
        best_model, best_score = None, -np.inf
        for seed in range(5):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=300,
                    random_state=seed * 42 + 7,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score, best_model = score, model
            except Exception:
                continue

        if best_model is not None:
            states = best_model.predict(X)
            probs = best_model.predict_proba(X)
            means = {
                s: float(stress.iloc[states == s].mean()) for s in range(n_states)
            }
            order = sorted(means, key=means.get)
            lmap = _build_label_map(order, n_states)
            regime = pd.Series([lmap[s] for s in states], index=idx)
            probs_df = pd.DataFrame(
                probs, index=idx, columns=[lmap[i] for i in range(n_states)]
            )
            return _smooth_regimes(regime, min_duration), probs_df, "HMM"

    if method in ("HMM", "GMM"):
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        gmm.fit(X)
        states = gmm.predict(X)
        probs = gmm.predict_proba(X)
        order = list(np.argsort(gmm.means_.flatten()))
        lmap = _build_label_map(order, n_states)
        regime = pd.Series([lmap[s] for s in states], index=idx)
        probs_df = pd.DataFrame(
            probs, index=idx, columns=[lmap[i] for i in range(n_states)]
        )
        tag = "GMM" if method == "GMM" else "GMM (hmmlearn missing)"
        return _smooth_regimes(regime, min_duration), probs_df, tag

    if n_states == 2:
        threshold = stress.quantile(0.50)
        labels = np.where(stress < threshold, "RISK_ON", "STRESS")
    else:
        q33, q67 = stress.quantile(0.33), stress.quantile(0.67)
        labels = np.select(
            [stress < q33, stress > q67],
            ["RISK_ON", "STRESS"],
            default="NEUTRAL",
        )
    regime = pd.Series(labels, index=idx)
    return _smooth_regimes(regime, min_duration), None, "Quantile"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_stress_index(raw: pd.Series, smooth: pd.Series, regime: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=raw.index, y=raw.values, mode="lines",
        name="Raw", line=dict(width=0.5, color="rgba(150,150,150,0.2)"),
    ))
    fig.add_trace(go.Scatter(
        x=smooth.index, y=smooth.values, mode="lines",
        name="Smoothed", line=dict(width=1.5, color="rgba(255,255,255,0.45)"),
    ))
    for label, color in REGIME_PALETTE.items():
        mask = regime == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=smooth.index[mask], y=smooth.loc[mask].values,
                mode="markers", name=label, marker=dict(color=color, size=3),
            ))
    return _base_layout(fig, "PCA Macro Stress Index", 420)


def plot_zscore_overlay(z_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in z_scores.columns:
        fig.add_trace(go.Scatter(
            x=z_scores.index, y=z_scores[col], name=col, line=dict(width=1),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    return _base_layout(fig, "Z-scored Stress Indicators", 350)


def plot_heatmap(z_scores: pd.DataFrame, days: int = 90) -> go.Figure:
    recent = z_scores.tail(days).T
    fig = go.Figure(go.Heatmap(
        z=recent.values,
        x=recent.columns.strftime("%Y-%m-%d"),
        y=recent.index,
        colorscale="RdYlGn_r",
        zmid=0,
    ))
    return _base_layout(fig, f"{days}-Day Stress Heatmap", 300)


def plot_loadings(loadings: pd.Series, explained: np.ndarray) -> go.Figure:
    ordered = loadings.sort_values()
    colors = ["#FF1744" if v > 0 else "#00C853" for v in ordered.values]
    fig = go.Figure(go.Bar(
        x=ordered.values, y=ordered.index, orientation="h", marker_color=colors,
    ))
    pct = explained[0] * 100
    return _base_layout(fig, f"PCA Loadings — PC1 = {pct:.1f}% variance", 280)


def plot_regime_probabilities(probs_df: pd.DataFrame | None) -> go.Figure | None:
    if probs_df is None:
        return None
    fig = go.Figure()
    for label in ("RISK_ON", "NEUTRAL", "STRESS"):
        if label in probs_df.columns:
            fig.add_trace(go.Scatter(
                x=probs_df.index, y=probs_df[label], name=label,
                stackgroup="one", line=dict(width=0),
                fillcolor=REGIME_PALETTE.get(label, "gray"),
            ))
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, "Regime Probabilities", 300)


def plot_spx_overlay(
    market_data: pd.DataFrame,
    stress: pd.Series,
    regime: pd.Series,
) -> go.Figure | None:
    if "SPX" not in market_data.columns:
        return None
    common_idx = stress.index.intersection(market_data.index)
    spx = market_data.loc[common_idx, "SPX"]
    reg = regime.loc[common_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.values, mode="lines",
        name="S&P 500", line=dict(width=1, color="rgba(200,200,200,0.25)"),
    ))
    for label, color in REGIME_PALETTE.items():
        mask = reg == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=spx.index[mask], y=spx.loc[mask].values,
                mode="markers", name=label, marker=dict(color=color, size=2),
            ))
    fig.update_yaxes(type="log")
    return _base_layout(fig, "S&P 500 — Regime Overlay", 380)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MRRM", layout="wide")
st.title("Macro Risk Regime Monitor")
st.caption("PCA · EMA-smoothed · HMM/GMM regime detection")

if not _HMM_AVAILABLE:
    st.info("HMM requires `pip install hmmlearn`. Using GMM fallback.")

with st.sidebar:
    st.header("Parameters")
    start_date = st.text_input("Start date", value="2007-01-01")
    lookback = st.slider("Z-score lookback (days)", 63, 756, 252, step=21)
    momentum_window = st.slider("Momentum window (days)", 5, 63, 21)
    vol_window = st.slider("Volatility window (days)", 10, 63, 21)
    clip_z = st.slider("Z-score clip", 1.0, 5.0, 3.0, step=0.5)
    ema_span = st.slider("EMA span (days)", 1, 42, 10)

    st.subheader("Regime detection")
    n_states = st.radio("States", [2, 3], index=1, horizontal=True)
    available_methods = ["HMM", "GMM", "Quantile"]
    default_method = 0 if _HMM_AVAILABLE else 1
    method = st.radio("Method", available_methods, index=default_method, horizontal=True)
    min_duration = st.slider("Min regime duration (days)", 1, 21, 5)

    if st.button("Clear cache"):
        st.cache_data.clear()

try:
    with st.spinner("Loading market data..."):
        market_data, fetch_errors = fetch_market_data(start_date)
    if fetch_errors:
        st.warning("Some tickers failed:\n" + "\n".join(fetch_errors))

    indicators = build_stress_indicators(market_data, momentum_window, vol_window)
    stress, stress_raw, z_scores, loadings, explained_var = compute_stress_index(
        indicators, lookback, clip_z, ema_span,
    )
    regime, regime_probs, method_tag = detect_regimes(
        stress, n_states, method, min_duration,
    )

    # Summary
    st.success(
        f"{stress.index[0].date()} → {stress.index[-1].date()} · "
        f"{len(stress):,} obs · {len(z_scores.columns)} factors · "
        f"PC1 = {explained_var[0] * 100:.1f}% · {method_tag}"
    )

    # Current state
    current_stress = stress.iloc[-1]
    current_regime = regime.iloc[-1]
    current_z = z_scores.iloc[-1]
    contributions = (current_z * loadings).sort_values(ascending=False)

    col_regime, col_stress = st.columns(2)
    col_regime.metric("Current regime", current_regime)
    col_stress.metric("Stress index", f"{current_stress:.2f}")

    z_cols = st.columns(len(z_scores.columns))
    for i, name in enumerate(z_scores.columns):
        z_cols[i].metric(f"Z({name})", f"{current_z[name]:.2f}")

    # Charts
    st.plotly_chart(plot_stress_index(stress_raw, stress, regime), width="stretch")
    st.plotly_chart(plot_zscore_overlay(z_scores), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_loadings(loadings, explained_var), width="stretch")
    with right:
        prob_fig = plot_regime_probabilities(regime_probs)
        if prob_fig:
            st.plotly_chart(prob_fig, width="stretch")
        else:
            st.info("Probabilities not available in Quantile mode.")

    st.plotly_chart(plot_heatmap(z_scores, days=90), width="stretch")

    spx_fig = plot_spx_overlay(market_data, stress, regime)
    if spx_fig:
        st.plotly_chart(spx_fig, width="stretch")

    # Decomposition table
    st.subheader("Stress decomposition (latest observation)")
    decomp = pd.DataFrame({
        "Z-score": current_z,
        "PC1 loading": loadings,
        "Contribution": contributions,
    }).sort_values("Contribution", ascending=False)
    st.dataframe(decomp.style.format("{:.3f}"), width="stretch")

    with st.expander("Explained variance by component"):
        var_table = pd.DataFrame({
            "Component": [f"PC{i + 1}" for i in range(len(explained_var))],
            "Variance (%)": explained_var * 100,
            "Cumulative (%)": np.cumsum(explained_var) * 100,
        })
        st.dataframe(
            var_table.style.format(
                {"Variance (%)": "{:.1f}", "Cumulative (%)": "{:.1f}"}
            ),
            width="stretch",
        )

    # Regime statistics
    st.subheader("Regime distribution")
    counts = regime.value_counts()
    pct = (counts / len(regime) * 100).round(1)
    st.dataframe(pd.DataFrame({"Count": counts, "%": pct}), width="stretch")

    # Event back-test
    with st.expander("Back-test against known events"):
        rows = []
        date_labels = stress.index.strftime("%Y-%m-%d")
        for date_str, event_name in REFERENCE_EVENTS.items():
            if date_str in date_labels.values:
                loc = stress.index[date_labels == date_str][0]
                rows.append({
                    "Date": date_str,
                    "Event": event_name,
                    "Stress": f"{stress.loc[loc]:.2f}",
                    "Regime": regime.loc[loc],
                    "Pass": "✓" if regime.loc[loc] == "STRESS" else "✗",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.caption("No reference events within the selected date range.")

    # CSV export
    export = pd.DataFrame({
        "stress_index": stress,
        "stress_raw": stress_raw,
        "regime": regime,
    })
    for col in z_scores.columns:
        export[f"z_{col}"] = z_scores[col]

    st.download_button(
        "Download CSV",
        export.to_csv().encode("utf-8"),
        "mrrm_export.csv",
        "text/csv",
    )

    with st.expander("Recent data"):
        st.dataframe(export.tail(10), width="stretch")

except Exception as exc:
    st.error(str(exc))
    import traceback

    st.code(traceback.format_exc())
