"""Streamlit dashboard for the exoplanet transit-detection pipeline.

All science lives in pipeline.py; this file is UI only.
Run with: streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline

# --- Page setup -------------------------------------------------------------
st.set_page_config(
    page_title="Exoplanet Detection AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background-color: #1e2532;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4B8BBE;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #fff; }
    .metric-label { font-size: 14px; color: #a0aebc; }
</style>
""",
    unsafe_allow_html=True,
)

SURFACE = "#0e1117"
ACCENT = "#4B8BBE"


def dark_fig(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    return fig, ax


# --- Sidebar ----------------------------------------------------------------
st.title("🪐 Exoplanet Detection AI")
st.markdown(
    "Enter a star designation to fetch real **TESS** data (SPOC lightcurves, with a "
    "TESScut **Full Frame Image** fallback), search for transits with TLS, run "
    "physics-based vetting, and score the candidate with a calibrated ML ensemble."
)

st.sidebar.header("Controls")
st.sidebar.markdown(
    """
### Example Targets
- **TOI-270** — known multi-planet system
- **TIC 307210830** — L 98-59 system
- **TIC 38846515** — known planet host
- **Ross 176** — no known transits
"""
)
target_star = st.sidebar.text_input("Enter Target Name", value="TOI-270")
max_sectors = st.sidebar.slider("Max sectors to analyze", 1, 5, pipeline.DEFAULT_MAX_SECTORS)
analyze_button = st.sidebar.button("Analyze System 🚀", type="primary")


@st.cache_resource
def load_model() -> tuple[dict | None, str | None]:
    return pipeline.load_model()


# --- Main flow --------------------------------------------------------------
if analyze_button and not target_star:
    st.warning("Please enter a target star name.")

if analyze_button and target_star:
    st.markdown("---")

    model_pkg, model_version = load_model()
    if model_pkg is None:
        st.error("Model not found. Run `python train_model.py` to create model/exoplanet_model_v3.pkl.")
        st.stop()

    progress = st.progress(5, text="Fetching lightcurve data...")
    with st.spinner(f"Querying MAST archive for {target_star}..."):
        lc, info = pipeline.load_lightcurve(target_star, max_sectors=max_sectors)
    if lc is None:
        progress.empty()
        st.error(f"Data fetch error: {info['message']}")
        st.stop()

    progress.progress(25, text="Looking up stellar parameters...")
    stellar = pipeline.get_stellar_params(target_star, dict(lc.meta))

    progress.progress(35, text="Detrending stellar variability...")
    time_arr, raw_flux, flat_flux, trend = pipeline.detrend(lc)
    t_bin, f_bin = pipeline.gap_aware_bin(time_arr, flat_flux)

    progress.progress(50, text="Running Transit Least Squares (this can take ~1 min)...")
    with st.spinner("Searching for periodic transit signals..."):
        tls_results = pipeline.run_tls(t_bin, f_bin, stellar=stellar)

    progress.progress(80, text="Extracting features and vetting...")
    period = pipeline.to_scalar(tls_results.period)
    duration = pipeline.transit_duration(tls_results)
    t0 = pipeline.to_scalar(tls_results.T0)
    sde = pipeline.to_scalar(tls_results.SDE)

    features = pipeline.extract_features(tls_results)
    depth_frac = features["depth_ppm"] * 1e-6
    symmetry, shape_ratio, depth_std = pipeline.calculate_shape_features(
        time_arr, flat_flux, period, duration, t0
    )
    oe_depth_diff, _, _, welch_p = pipeline.odd_even_test(time_arr, flat_flux, period, duration, t0)
    oe_rel_diff = oe_depth_diff / depth_frac if depth_frac > 0 else 0.0
    odd_even_ok = welch_p >= pipeline.WELCH_P_THRESHOLD or oe_rel_diff < pipeline.OE_REL_DIFF_THRESHOLD
    physics = pipeline.check_transit_physics(period, duration, stellar["radius"], stellar["mass"])
    secondary = pipeline.check_secondary_eclipse(
        time_arr, flat_flux, period, duration, t0, primary_depth_frac=depth_frac
    )

    progress.progress(95, text="Scoring with ML model...")
    pred = pipeline.predict(model_pkg, features)
    progress.progress(100, text="Complete!")
    progress.empty()

    # --- Results ------------------------------------------------------------
    st.header("Prediction Results")
    st.caption(
        f"Data source: **{info['source']}** · {info['n_sectors']} sector(s) · "
        f"{len(time_arr):,} points ({len(t_bin):,} after binning) · "
        f"Stellar params: R={stellar['radius']:.2f} R☉, M={stellar['mass']:.2f} M☉ "
        f"({stellar['source']}) · Model {model_version}"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        color = "#4CAF50" if pred["prediction"] == 1 else "#F44336"
        st.markdown(
            f"""<div class="metric-card" style="border-left-color: {color};">
            <div class="metric-label">Model Classification</div>
            <div class="metric-value">{pred['result_text']}</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card" style="border-left-color: #FFC107;">
            <div class="metric-label">Planet Probability</div>
            <div class="metric-value">{pred['probability']:.2%}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card" style="border-left-color: #2196F3;">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value">{pred['confidence']}</div></div>""",
            unsafe_allow_html=True,
        )

    # --- Physics vetting ----------------------------------------------------
    vet1, vet2 = st.columns(2)
    with vet1:
        if sde >= pipeline.SDE_THRESHOLD:
            st.success(f"✅ **Signal Detection Passed:** SDE = {sde:.1f} ≥ {pipeline.SDE_THRESHOLD:.0f}.")
        else:
            st.warning(
                f"⚠️ **Weak Signal:** SDE = {sde:.1f} < {pipeline.SDE_THRESHOLD:.0f}. "
                "The periodic signal is not statistically significant."
            )
    with vet2:
        if not odd_even_ok:
            st.warning(
                f"⚠️ **Odd-Even Depth Alert:** odd and even transits differ significantly "
                f"(Welch p = {welch_p:.2e}, {oe_rel_diff:.0%} of transit depth). "
                "Classic eclipsing-binary signature."
            )
        else:
            st.success(
                f"✅ **Odd-Even Depth Passed:** consistent depths "
                f"(p = {welch_p:.3f}, relative diff {oe_rel_diff:.1%})."
            )

    vet3, vet4 = st.columns(2)
    with vet3:
        if not physics["duration_ok"]:
            st.warning(
                f"⚠️ **Transit Physics Alert:** duration is {physics['duration_ratio']:.2f}× the "
                "maximum circular-orbit duration — physically anomalous."
            )
        elif not physics["density_ok"]:
            st.warning(
                f"⚠️ **Stellar Density Alert:** transit-implied density is "
                f"{physics['density_ratio']:.2f}× the catalog stellar density — likely a blend "
                "or eclipsing binary."
            )
        else:
            st.success(
                f"✅ **Transit Physics Passed:** duration {physics['duration_ratio']:.2f}× of circular "
                f"limit; density ratio {physics['density_ratio']:.2f}."
            )
    with vet4:
        if secondary["has_secondary"]:
            st.warning(
                f"⚠️ **Secondary Eclipse Alert:** dip at phase 0.5 "
                f"(depth = {secondary['secondary_depth']:.5f}, S/N = {secondary['secondary_snr']:.1f}). "
                "Likely a stellar companion."
            )
        else:
            st.success(
                f"✅ **Secondary Eclipse Passed:** no significant dip at phase 0.5 "
                f"(S/N = {secondary['secondary_snr']:.1f} < {pipeline.SECONDARY_SNR_THRESHOLD:.0f})."
            )

    # --- Verdict reasoning ---------------------------------------------------
    explanation = pipeline.build_explanation(
        {
            "data": {"source": info["source"], "n_sectors": info["n_sectors"],
                     "n_points": len(time_arr), "n_points_binned": len(t_bin)},
            "stellar": stellar,
            "features": features,
            "diagnostics": {"sde": sde, "sde_pass": sde >= pipeline.SDE_THRESHOLD,
                            "welch_p": welch_p, "odd_even_rel_diff": oe_rel_diff},
            "vetting": {**physics, **secondary, "odd_even_ok": odd_even_ok},
            "prediction": pred,
        }
    )
    st.markdown("---")
    st.subheader("Why This Verdict")
    for line in explanation[:-1]:
        st.markdown(f"- {line}")
    verdict_line = explanation[-1]
    if pred["prediction"] == 1:
        st.success(f"**{verdict_line}**")
    elif "AMBIGUOUS" in verdict_line:
        st.warning(f"**{verdict_line}**")
    else:
        st.error(f"**{verdict_line}**")

    # --- Plots --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Lightcurve Visualizations")
    tab1, tab2, tab3 = st.tabs(["Raw & Detrended Flux", "Phase-Folded Transit", "TLS Periodogram"])

    with tab1:
        fig1, ax1 = dark_fig((12, 4))
        ax1.plot(time_arr, raw_flux, ".", color="gray", alpha=0.3, markersize=2, label="Raw flux")
        ax1.plot(time_arr, trend, "-", color="red", alpha=0.8, linewidth=1, label="Trend")
        ax1.plot(time_arr, flat_flux - 0.02, ".", color=ACCENT, alpha=0.4, markersize=2,
                 label="Detrended (−0.02 offset)")
        ax1.set_xlabel("Time (BTJD days)")
        ax1.set_ylabel("Normalized flux")
        ax1.legend(loc="upper right", facecolor="#1e2532", labelcolor="white")
        st.pyplot(fig1)

    with tab2:
        if hasattr(tls_results, "folded_phase") and period > 0:
            fig2, ax2 = dark_fig((10, 5))
            ax2.plot(tls_results.folded_phase, tls_results.folded_y, ".", color="gray",
                     alpha=0.3, markersize=2, zorder=1)
            if hasattr(tls_results, "model_folded_phase"):
                ax2.plot(tls_results.model_folded_phase, tls_results.model_folded_model,
                         color="red", linewidth=2, zorder=2, label="TLS model")
                ax2.legend(loc="lower right", facecolor="#1e2532", labelcolor="white")
            window = max(2.5 * duration / period, 0.02)
            ax2.set_xlim(0.5 - window, 0.5 + window)
            ax2.set_xlabel("Orbital phase")
            ax2.set_ylabel("Relative flux")
            ax2.set_title(f"Phase-folded at P = {period:.4f} d")
            st.pyplot(fig2)
        else:
            st.info("No valid transit signal to fold.")

    with tab3:
        if hasattr(tls_results, "periods"):
            fig3, ax3 = dark_fig((10, 4))
            ax3.plot(tls_results.periods, tls_results.power, color=ACCENT, linewidth=0.8)
            if period > 0:
                ax3.axvline(period, color="red", linestyle="--", alpha=0.8)
            ax3.axhline(pipeline.SDE_THRESHOLD, color="orange", linestyle=":", alpha=0.6)
            ax3.set_xlabel("Trial period (days)")
            ax3.set_ylabel("SDE")
            st.pyplot(fig3)
        else:
            st.info("Periodogram unavailable.")

    # --- Feature tables -----------------------------------------------------
    st.markdown("---")
    st.subheader("Model Features & Diagnostics")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.dataframe(
            pd.DataFrame(
                {"Feature": list(features.keys()),
                 "Value": [f"{v:.5g}" for v in features.values()]}
            ),
            use_container_width=True, hide_index=True,
        )
    with col_f2:
        diagnostics = {
            "SDE": sde,
            "T0 (BTJD)": t0,
            "Transit symmetry (ingress/egress)": symmetry,
            "Shape ratio": shape_ratio,
            "In-transit scatter": depth_std,
            "Odd-even Welch p-value": welch_p,
            "Secondary eclipse S/N": secondary["secondary_snr"],
            "Density ratio (transit/catalog)": physics["density_ratio"],
        }
        st.dataframe(
            pd.DataFrame(
                {"Diagnostic": list(diagnostics.keys()),
                 "Value": [f"{v:.5g}" for v in diagnostics.values()]}
            ),
            use_container_width=True, hide_index=True,
        )
