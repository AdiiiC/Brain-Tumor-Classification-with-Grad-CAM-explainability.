"""Pydantic schemas for API request/response models."""

from datetime import datetime

from pydantic import BaseModel, Field


class BuildInfo(BaseModel):
    api_version: str
    model_version: str
    git_sha: str


class OODInfo(BaseModel):
    is_out_of_distribution: bool
    score: float
    mahalanobis_distance: float | None = None
    energy: float
    method: str
    message: str


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    uncertainty: float | None = None
    calibrated_confidence: float | None = None
    flagged_for_review: bool = False
    out_of_distribution: OODInfo | None = None
    study_id: str | None = None
    model_version: str | None = None
    git_sha: str | None = None


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
    calibration_temperature: float
    classes: list[str]
    api_version: str
    model_version: str
    git_sha: str
    ood_detector_fitted: bool
    segmentation_model: str
    database_connected: bool
    auth_required: bool
    explainability_available: bool = True

    model_config = {"protected_namespaces": ()}


class Measurements(BaseModel):
    area_pixels: int
    area_mm2: float
    volume_mm3: float
    volume_cm3: float
    max_diameter_mm: float
    pixel_spacing_mm: float
    slice_thickness_mm: float


class SegmentationResult(BaseModel):
    method: str
    approximate: bool
    spacing_estimated: bool
    measurements: Measurements
    mask_base64: str
    overlay_base64: str
    note: str
    predicted_class: str | None = None
    confidence: float | None = None
    study_id: str | None = None


class FeedbackRequest(BaseModel):
    corrected_class: str = Field(..., description="Class confirmed by the reviewing radiologist")
    reviewer: str | None = Field(None, max_length=128)
    notes: str | None = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    study_id: str
    corrected_class: str
    agrees_with_ai: bool
    reviewer: str | None = None
    recorded_at: datetime


class StudySummary(BaseModel):
    id: str
    created_at: datetime
    patient_id: str | None = None
    filename: str | None = None
    predicted_class: str
    confidence: float
    uncertainty: float | None = None
    tumor_volume_mm3: float | None = None
    quality_score: float | None = None
    sequence_type: str | None = None
    who_grade: str | None = None
    is_ood: bool
    flagged_for_review: bool
    model_version: str
    confirmed_class: str | None = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}
