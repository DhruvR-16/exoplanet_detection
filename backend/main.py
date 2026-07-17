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

    pred = result["prediction"]
    arrays = result["arrays"]
    time_arr = arrays["time"]

    # Downsample plot data for web transfer
    step = max(1, len(time_arr) // 1000)
    plot_data = {
        "time": time_arr[::step].tolist(),
        "raw_flux": arrays["raw_flux"][::step].tolist(),
        "flat_flux": arrays["flat_flux"][::step].tolist(),
        "trend": arrays["trend"][::step].tolist(),
    }

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
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
