# Direct Python export of main.ipynb. Code cells are unchanged.

# %% [markdown]
# # 🪐 TESS Exoplanet Detection Pipeline
#
# End-to-end demo of the detection pipeline in [`pipeline.py`](pipeline.py):
#
# 1. **Fetch** — TESS SPOC lightcurves from MAST (TESScut *Full Frame Image* fallback), stitched across sectors and cached locally.
# 2. **Detrend** — `wotan` biweight filter removes stellar variability; gap-aware 10-minute median binning speeds up the search.
# 3. **Search** — Transit Least Squares (TLS) with stellar priors from the TIC catalog.
# 4. **Vet** — physics checks: SDE threshold, odd/even depth (Welch's t-test), transit duration & stellar-density consistency, secondary-eclipse search.
# 5. **Classify** — calibrated RandomForest + XGBoost ensemble trained on the NASA Exoplanet Archive KOI catalog (`train_model.py`).
#
# > Change `TARGET` below and re-run to analyze any TESS star.

# %%
import warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np

import pipeline

TARGET = "TOI-270"   # try: "TIC 307210830", "TIC 38846515", "Ross 176"

FIG_DIR = pipeline.ROOT_DIR / "docs" / "img"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Dark figure theme, consistent with the docs/ figures
SURFACE, PANEL = "#0d1117", "#161b22"
INK, MUTED, GRID = "#e6edf3", "#8b949e", "#21262d"
ACCENT, RED, AMBER = "#818cf8", "#f87171", "#f59e0b"

def dark_ax(ax, title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=INK, fontweight="bold", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)

def dark_legend(ax, loc="lower right"):
    leg = ax.legend(loc=loc, facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9)
    leg.get_frame().set_alpha(0.9)

# %% [markdown]
# ## 1 · Load the trained model

# %%
model_pkg, model_version = pipeline.load_model()
assert model_pkg is not None, "No model found - run `python train_model.py` first"

m = model_pkg["metrics"]
print(f"Model {model_version}: {model_pkg['description']}")
print(f"Held-out test set: accuracy={m['accuracy']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
      f"PR-AUC={m['pr_auc']:.3f}  (5-fold CV AUC {m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f})")

# %% [markdown]
# ## 2 · Fetch TESS data and stellar parameters

# %%
lc, info = pipeline.load_lightcurve(TARGET)
assert lc is not None, info["message"]

stellar = pipeline.get_stellar_params(TARGET, dict(lc.meta))
print(f"{TARGET}: source={info['source']}, {info['n_sectors']} sector(s), {len(lc):,} cadences")
print(f"Star: R = {stellar['radius']:.3f} R_sun, M = {stellar['mass']:.3f} M_sun  ({stellar['source']})")

# %% [markdown]
# ## 3 · Detrend stellar variability and bin

# %%
time_arr, raw_flux, flat_flux, trend = pipeline.detrend(lc)
t_bin, f_bin = pipeline.gap_aware_bin(time_arr, flat_flux)
print(f"{len(time_arr):,} points -> {len(t_bin):,} after gap-aware 10-min binning")

fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=140, facecolor=SURFACE, sharex=True)
axes[0].plot(time_arr, raw_flux, ".", ms=1.5, color=MUTED, alpha=0.5, label="Raw flux")
axes[0].plot(time_arr, trend, "-", color=RED, lw=1.2, label="Wotan biweight trend")
axes[1].plot(time_arr, flat_flux, ".", ms=1.5, color=ACCENT, alpha=0.5, label="Detrended flux")
dark_ax(axes[0], f"{TARGET} — raw flux and stellar-variability trend")
dark_ax(axes[1])
axes[1].set_xlabel("Time (BTJD days)")
for ax in axes:
    ax.set_ylabel("Normalized flux")
    dark_legend(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "pipeline_detrend.png", bbox_inches="tight", facecolor=SURFACE)
plt.show()

# %% [markdown]
# ## 4 · Transit Least Squares search

# %%
tls_results = pipeline.run_tls(t_bin, f_bin, stellar=stellar)

period = pipeline.to_scalar(tls_results.period)
duration = pipeline.transit_duration(tls_results)
t0 = pipeline.to_scalar(tls_results.T0)
sde = pipeline.to_scalar(tls_results.SDE)
print(f"Best period: {period:.5f} d | duration: {duration * 24:.2f} h | SDE: {sde:.1f}")

# %%
fig, ax = plt.subplots(figsize=(11, 4), dpi=140, facecolor=SURFACE)
ax.plot(tls_results.periods, tls_results.power, color=ACCENT, lw=0.9)
ax.axvline(period, color=RED, ls="--", alpha=0.8)
ax.axhline(pipeline.SDE_THRESHOLD, color=AMBER, ls=":", alpha=0.7)
ax.annotate(f"P = {period:.3f} d", xy=(period * 1.06, sde * 0.94), color=INK, fontsize=10, fontweight="bold")
dark_ax(ax, f"{TARGET} — TLS periodogram")
ax.set_xlabel("Trial period (days)")
ax.set_ylabel("SDE")
fig.tight_layout()
fig.savefig(FIG_DIR / "pipeline_periodogram.png", bbox_inches="tight", facecolor=SURFACE)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=140, facecolor=SURFACE)
ax.plot(tls_results.folded_phase, tls_results.folded_y, ".", ms=2.5, color=MUTED, alpha=0.45,
        label="Folded flux")
ax.plot(tls_results.model_folded_phase, tls_results.model_folded_model, color=RED, lw=2,
        label="TLS transit model")
window = max(2.5 * duration / period, 0.02)
ax.set_xlim(0.5 - window, 0.5 + window)
dark_ax(ax, f"{TARGET} — phase-folded transit (P = {period:.4f} d)")
ax.set_xlabel("Orbital phase")
ax.set_ylabel("Relative flux")
dark_legend(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "pipeline_phase_fold.png", bbox_inches="tight", facecolor=SURFACE)
plt.show()

# %% [markdown]
# ## 5 · Physics vetting and ML classification

# %%
result = pipeline.analyze(TARGET, model_pkg=model_pkg)

print("Model features:")
for name, value in result["features"].items():
    print(f"  {name:>22}: {value:.5g}")

print("\nVerdict reasoning:")
for line in result["explanation"]:
    print("  -", line)
