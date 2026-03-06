from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import io

from app.core.model import load_artifacts
from app.schemas.predict import PredictRequest, PredictResponse
from app.db.database import SessionLocal
from app.db.models import PredictionRequest, PredictionOutput

app = FastAPI(
    title="Turnover Attrition Prediction API",
    description="FastAPI service exposing a trained ML model to predict employee attrition.",
    version="0.2.0",
)

artifacts = None


@app.on_event("startup")
def startup_event():
    global artifacts
    artifacts = load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    global artifacts
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    required = artifacts.metrics.get("raw_feature_names", [])
    missing = [c for c in required if c not in payload.features]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}"
        )

    db = SessionLocal()

    try:
        # Enregistrer l'input dans PostgreSQL
        request_row = PredictionRequest(
            source="api_predict",
            payload=payload.features
        )
        db.add(request_row)
        db.commit()
        db.refresh(request_row)

        # Faire la prédiction
        X = pd.DataFrame([payload.features])[required]

        proba = artifacts.pipeline.predict_proba(X)[:, 1][0]
        pred = int(proba >= artifacts.threshold)

        response_data = {
            "probability": float(proba),
            "prediction": pred,
            "threshold": float(artifacts.threshold),
            "model_info": {
                "model": artifacts.metrics.get("model"),
                "test_average_precision": artifacts.metrics.get("test_average_precision"),
            } if artifacts.metrics else None,
        }

        # Enregistrer l'output dans PostgreSQL
        output_row = PredictionOutput(
            request_id=request_row.id,
            probability=float(proba),
            prediction=pred,
            threshold=float(artifacts.threshold),
            model_name=artifacts.metrics.get("model"),
            test_average_precision=artifacts.metrics.get("test_average_precision"),
            response_payload=response_data,
        )
        db.add(output_row)
        db.commit()

        # Retourner la réponse API
        return PredictResponse(**response_data)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    finally:
        db.close()


@app.get("/schema")
def schema():
    global artifacts
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = artifacts.metrics.get("raw_feature_names", [])
    categorical_levels = artifacts.metrics.get("categorical_levels", {})

    schema_dict = {}
    for f in features:
        if f in categorical_levels:
            schema_dict[f] = {
                "type": "categorical",
                "example_values": categorical_levels[f],
            }
        else:
            schema_dict[f] = {"type": "numeric"}

    return {
        "n_features": len(features),
        "features": schema_dict,
    }


@app.get("/example")
def example():
    global artifacts
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {"features": artifacts.metrics.get("example_features")}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    global artifacts

    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")

    required = artifacts.metrics.get("raw_feature_names", [])
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing}"
        )

    X = df[required]

    try:
        proba = artifacts.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= artifacts.threshold).astype(int)

        out = df.copy()
        out["probability"] = proba
        out["prediction"] = preds

        return out.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))