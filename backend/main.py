"""FastAPI backend for the exoplanet detection pipeline.

Thin HTTP layer over pipeline.py. Run from the repo root or backend/:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline  # noqa: E402  (needs repo root on sys.path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Exoplanet Detection API", version="3.0.0")

# Dev frontend origins; extend via comma-separated ALLOWED_ORIGINS env var.
allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MODEL_PKG, MODEL_VERSION = pipeline.load_model()
if MODEL_PKG is None:
    logger.error("No model found in %s — run `python train_model.py` first.", pipeline.MODEL_DIR)


class AnalysisRequest(BaseModel):
    target_star: str = Field(min_length=1, max_length=100)
    max_sectors: int = Field(default=pipeline.DEFAULT_MAX_SECTORS, ge=1, le=5)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_version": MODEL_VERSION, "model_loaded": MODEL_PKG is not None}


@app.get("/api/targets/examples")
def get_example_targets() -> list[dict[str, str]]:
    return [
        {
            "name": "TOI-270",
            "type": "Known Multi-Planet System",
            "description": "Bright M-dwarf host to three confirmed transiting planets (TOI-270 b, c, d). Excellent high-SNR candidate.",
            "expected": "Planet Candidate (High Confidence)"
        },
        {
            "name": "TIC 307210830",
            "type": "L 98-59 System",
            "description": "Nearby M dwarf hosting multiple terrestrial exoplanets discovered by TESS.",
            "expected": "Planet Candidate"
        },
        {
            "name": "TIC 38846515",
            "type": "Transiting Exoplanet Host",
            "description": "Known planet-hosting target star analyzed across multiple TESS sectors.",
            "expected": "Planet Candidate"
        },
        {
            "name": "Ross 176",
            "type": "Quiet Control Star",
            "description": "Nearby M-dwarf target star without known transits; useful for baseline testing.",
            "expected": "No Planet Signal Detected"
        }
    ]


@app.post("/api/analyze")
def analyze_target(req: AnalysisRequest) -> dict[str, Any]:
    if MODEL_PKG is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run `python train_model.py` first.")

    try:
        result = pipeline.analyze(req.target_star, model_pkg=MODEL_PKG, max_sectors=req.max_sectors)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline failed for %s", req.target_star)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    pred = result.get("prediction") or {
        "prediction": 0,
        "probability": 0.5,
        "confidence": "Low",
        "result_text": "No Model Prediction",
    }
    arrays = result["arrays"]
    time_arr = arrays["time"]
    tls = result["tls"]

    # 1. Full Light Curve Plot Data
    step = max(1, len(time_arr) // 1000)
    plot_data = {
        "time": time_arr[::step].tolist(),
        "raw_flux": arrays["raw_flux"][::step].tolist(),
        "flat_flux": arrays["flat_flux"][::step].tolist(),
        "trend": arrays["trend"][::step].tolist(),
    }

    # 2. Phase-Folded Transit Data
    folded_data = {"phase": [], "flux": [], "model_phase": [], "model_flux": []}
    try:
        f_phase = getattr(tls, "folded_phase", np.array([]))
        f_y = getattr(tls, "folded_y", np.array([]))
        m_phase = getattr(tls, "model_folded_phase", np.array([]))
        m_y = getattr(tls, "model_folded_model", np.array([]))

        if len(f_phase) > 0 and len(f_y) > 0:
            f_step = max(1, len(f_phase) // 600)
            folded_data["phase"] = f_phase[::f_step].tolist()
            folded_data["flux"] = f_y[::f_step].tolist()

        if len(m_phase) > 0 and len(m_y) > 0:
            m_step = max(1, len(m_phase) // 300)
            folded_data["model_phase"] = m_phase[::m_step].tolist()
            folded_data["model_flux"] = m_y[::m_step].tolist()
    except Exception as e:
        logger.warning("Could not extract folded transit data: %s", e)

    # 3. Periodogram Data
    periodogram_data = {"periods": [], "sde": []}
    try:
        periods = getattr(tls, "periods", np.array([]))
        power = getattr(tls, "power", np.array([]))
        if len(power) == 0:
            power = getattr(tls, "SDE", np.array([]))

        if len(periods) > 0 and len(power) > 0:
            p_step = max(1, len(periods) // 600)
            periodogram_data["periods"] = periods[::p_step].tolist()
            periodogram_data["sde"] = power[::p_step].tolist()
    except Exception as e:
        logger.warning("Could not extract periodogram data: %s", e)

    # 4. Odd vs Even Transits Data
    odd_even_data = {"odd_phase": [], "odd_flux": [], "even_phase": [], "even_flux": []}
    try:
        period = float(result["features"].get("period_days", 0.0))
        t0 = float(result["diagnostics"].get("t0", 0.0))
        if period > 0:
            phase = ((time_arr - t0) % period) / period
            phase[phase > 0.5] -= 1.0
            transit_num = np.round((time_arr - t0) / period)

            in_range = np.abs(phase) <= 0.15
            odd_mask = in_range & (transit_num % 2 == 1)
            even_mask = in_range & (transit_num % 2 == 0)

            odd_step = max(1, int(odd_mask.sum()) // 300)
            even_step = max(1, int(even_mask.sum()) // 300)

            odd_even_data["odd_phase"] = phase[odd_mask][::odd_step].tolist()
            odd_even_data["odd_flux"] = arrays["flat_flux"][odd_mask][::odd_step].tolist()
            even_phase_arr = phase[even_mask][::even_step]
            even_flux_arr = arrays["flat_flux"][even_mask][::even_step]
            odd_even_data["even_phase"] = even_phase_arr.tolist()
            odd_even_data["even_flux"] = even_flux_arr.tolist()
    except Exception as e:
        logger.warning("Could not extract odd/even data: %s", e)

    vetting = result["vetting"]
    diagnostics = result["diagnostics"]
    return {
        "target": result["target"],
        "predictions": [
            {
                "model_name": f"Calibrated RF+XGBoost Ensemble ({MODEL_VERSION})",
                "prediction": pred["prediction"],
                "probability": pred["probability"],
                "confidence": pred["confidence"],
                "result_text": pred["result_text"],
            }
        ],
        "features": result["features"],
        "explanation": result["explanation"],
        "data_source": result["data"]["source"],
        "n_sectors": result["data"]["n_sectors"],
        "sde": diagnostics["sde"],
        "sde_pass": diagnostics["sde_pass"],
        "welch_p": diagnostics["welch_p"],
        "duration_ok": vetting["duration_ok"],
        "duration_ratio": vetting["duration_ratio"],
        "density_ok": vetting["density_ok"],
        "density_ratio": vetting["density_ratio"],
        "has_secondary": vetting["has_secondary"],
        "secondary_depth": vetting["secondary_depth"],
        "secondary_snr": vetting["secondary_snr"],
        "stellar_r": result["stellar"]["radius"],
        "stellar_m": result["stellar"]["mass"],
        "plot_data": plot_data,
        "folded_data": folded_data,
        "periodogram_data": periodogram_data,
        "odd_even_data": odd_even_data,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

