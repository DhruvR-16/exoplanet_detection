# Research study: characterizing the pipeline

This package turns the working pipeline ([`../pipeline.py`](../pipeline.py)) into
a **characterized decision system** for the accompanying paper
([`../paper/`](../paper/)). Nothing here re-implements the science — every module
imports the pipeline and studies its behavior.

> **Thesis.** A transit-detection pipeline should be reported not as a single
> accuracy number but as (1) a measured detection *completeness*, (2) an honest
> *cross-mission* transfer result, and (3) a *disagreement* signal between the
> learned score and independent physics vetting — which is where the
> follow-up-worthy candidates live.

## Modules

| Module | Contribution | Output |
| :--- | :--- | :--- |
| `tess_benchmark.py` | Real TESS labels (ExoFOP CP/KP vs FP/FA) → cross-mission metrics | `results/tess_benchmark.parquet`, `docs/img/tess_*.png` |
| `injection.py` | Mandel–Agol injection–recovery into screened-quiet TESS hosts | `results/injection.parquet`, `docs/img/injection_*.png` |
| `disagreement.py` | ML × physics 2×2 triage quadrants + purity | `docs/img/disagreement_*.png`, `results/disagreement_*.csv` |
| `baselines.py` | Model comparison + feature/physics ablations (cross-mission) | `results/baselines_*.csv`, `docs/img/baselines_auc.png` |
| `cnn_baseline.py` | 1D-CNN on parametric folded views (shape-vs-scalar) | `results/cnn_baseline.json`, `docs/img/cnn_views.png` |
| `statistics.py` | Bootstrap AUC CIs, Wilson purity CIs, paired sector deltas | `results/statistics.json`, `docs/img/quadrant_purity_ci.png` |
| `triage.py` | Follow-up efficiency: physics-informed vs ML-probability ranking | `results/triage_summary.json`, `docs/img/triage_efficiency.png` |
| `error_analysis.py` | Failure modes + operating-point diagnosis and fix | `results/error_analysis.json`, `docs/img/error_analysis.png` |
| `adaptation.py` | Physics-anchored self-training (label-free domain adaptation) | `results/adaptation.json`, `docs/img/adaptation.png` |
| `run_all.py` | Orchestrates the whole study | all of the above |
| `theme.py` | Shared dark-mode figure styling | — |

## Reproduce

```bash
pip install -r requirements.txt          # adds batman-package, pyarrow
python -m research.run_all --smoke        # fast sanity pass (small samples)
python -m research.run_all                # full study (hours; heavy TLS + downloads)
python -m research.run_all --figures-only # redraw figures from existing results
```

Heavy stages (`tess_benchmark`, `injection`) are **resumable** — an interrupted
run continues from its CSV log.

## Design notes / honest caveats

- **Single-sector** TESS light curves are used for tractability; this
  lower-bounds detection performance (multi-sector would raise completeness). The
  benchmark's `--max-sectors` flag lifts this for a fuller run.
- **Injection hosts are screened** (pre-injection TLS SDE < 8) so the injected
  transit is the only signal; real correlated noise still varies across the sky.
- The **1D-CNN** trains on parametric folded views generated with `batman`, not
  raw pixels — a controlled shape-vs-scalar comparison, with a full AstroNet-style
  detector left as future work.
- Feature **units match** the KOI training catalog and the TLS inference outputs,
  so there is no train/serve skew.
