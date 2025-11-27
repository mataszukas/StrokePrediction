# ...existing code...
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import dill, json, os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import logging
import threading
from contextlib import asynccontextmanager

MODEL_PATH = os.path.join("models", "lsvc_calibrated.dill")
META_PATH = os.path.join("models", "lsvc_calibrated_metadata.json")
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("application startup begin")
    try:
        # start model load in background so startup doesn't block the container warmup
        t = threading.Thread(target=load_model_and_meta, name="model-loader", daemon=True)
        t.start()
    except Exception:
        logging.exception("Error starting background model loader")
    logging.info("application startup complete")
    yield
    logging.info("application shutdown complete")

app = FastAPI(title="stroke-model", lifespan=lifespan)

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

model = None
meta = {}
model_lock = threading.Lock()   # prevent concurrent loads

def download_model_if_missing():
    Path("models").mkdir(parents=True, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return
    url = os.environ.get("MODEL_BLOB_URL")
    if not url:
        logging.info("MODEL_BLOB_URL not set; skipping model download")
        return
    try:
        logging.info("Downloading model from %s", url)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        logging.info("Model downloaded to %s", MODEL_PATH)
    except Exception:
        logging.exception("Failed to download model from MODEL_BLOB_URL")

def load_model_and_meta():
    global model, meta
    # avoid duplicate concurrent loads
    with model_lock:
        if model is not None:
            return

        # ensure model is present (download if necessary)
        download_model_if_missing()

        # load model (handle missing file / errors)
        try:
            with open(MODEL_PATH, "rb") as f:
                model = dill.load(f)
                logging.info("Model loaded from %s", MODEL_PATH)
        except FileNotFoundError:
            logging.warning("Model file not found: %s", MODEL_PATH)
            model = None
        except Exception:
            logging.exception("Failed to load model; leaving model as None")
            model = None

        # load metadata
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
                logging.info("Metadata loaded from %s", META_PATH)
        except FileNotFoundError:
            meta = {}
        except Exception:
            logging.exception("Failed to load metadata; using empty meta")
            meta = {}

def align_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    features = meta.get("features") or []
    if features:
        for c in features:
            if c not in df.columns:
                df[c] = np.nan
        return df[features]
    return df

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": meta.get("model_version"),
        "n_features": len(meta.get("features", []))
    }

@app.post("/predict")
def predict(req: PredictRequest):
    global model, meta
    # avoid logging full payload; log count instead
    try:
        n_records = len(req.records)
    except Exception:
        n_records = 0
    logging.debug("Predict request n=%d", n_records)

    if model is None:
        load_model_and_meta()
    if model is None:
        # model still not available
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.records:
        raise HTTPException(status_code=400, detail="no records provided")

    df = pd.DataFrame(req.records)

    # server-side coercion of known numeric columns
    numeric_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    X = align_features(df.copy(), meta)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            probs_list = probs.tolist()
            checks = ["Yes" if p > 0.062 else "No" for p in probs_list]
            result = {"Probability of Stroke": probs.tolist(), "Threshold": 0.062, "Regular Checks Needed?": checks}
            logging.info("Predict response: n=%d", len(probs))
            return result
        preds = model.predict(X)
        logging.info("Predict response: n=%d", len(preds))
        return {"Predictions": preds.tolist()}
    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-model")
def load_model_endpoint():
    try:
        load_model_and_meta()
        return {"status": "ok", "model_loaded": model is not None}
    except Exception as e:
        logging.exception("load-model error")
        raise HTTPException(status_code=500, detail=str(e))

# serve the UI at both root and /ui
@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def index():
    try:
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception:
        logging.exception("Failed to read template")
        raise HTTPException(status_code=500, detail="template read error")