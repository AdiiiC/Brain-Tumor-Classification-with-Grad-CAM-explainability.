"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    uncertainty: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    flagged_for_review: bool = False


class BatchPredictionResult(BaseModel):
    filename: str
    result: PredictionResult


class GradCAMResult(BaseModel):
    predicted_class: str
    confidence: float
    heatmap_base64: str
    overlay_base64: str


class SHAPResult(BaseModel):
    predicted_class: str
    confidence: float
    shap_image_base64: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    calibration_enabled: bool
    classes: list[str]
