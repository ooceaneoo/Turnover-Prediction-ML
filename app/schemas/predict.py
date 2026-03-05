from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary used for prediction")


class PredictResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="0 or 1")
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_info: Optional[Dict[str, Any]] = Field(None, description="Optional model metrics/info")