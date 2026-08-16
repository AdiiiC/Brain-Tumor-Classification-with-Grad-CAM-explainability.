"""
FastAPI Backend — BrainScan AI.

Serves the brain tumor classification model via REST.
Doctors upload MRI scans → prediction + Grad-CAM++ + uncertainty + volumetry.

Includes:
- API-key auth, per-endpoint rate limiting, content-based upload validation
- Out-of-distribution gating before any clinical result is reported
- Persistent studies, radiologist feedback, longitudinal tracking
- PDF report export
- Structured logging and /metrics
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import config
from api.database import Feedback, Study, get_session, image_fingerprint, init_db
from api.dicom_handler import dicom_to_image, extract_dicom_metadata, is_dicom
from api.image_quality import ImageQualityAssessor
from api.longitudinal_service import analyze_patient
from api.model_service import CLASS_NAMES, ModelService
from api.observability import ObservabilityMiddleware, configure_logging, log_event, metrics
from api.onnx_service import ONNX_AVAILABLE, OnnxModelService
from api.ood import OODDetector
from api.pediatric_support import PediatricAssessor
from api.schemas import (
    BatchPredictionResult,
    FeedbackRequest,
    FeedbackResponse,
    GradCAMResult,
    HealthResponse,
    OODInfo,
    PredictionResult,
    SegmentationResult,
    SHAPResult,
    StudySummary,
)
from api.security import (
    auth_enabled,
    batch_rate_limit,
    heavy_rate_limit,
    require_api_key,
    standard_rate_limit,
)
from api.segmentation_service import SegmentationService
from api.sequence_detector import SequenceDetector
from api.small_tumor_detector import SmallTumorDetector
from api.tumor_grading import TumorGrader
from api.validation import decode_image, read_validated

MODEL_PATH = os.getenv("MODEL_PATH", "model_best.keras")
TFLITE_PATH = os.getenv("TFLITE_PATH", "brain_tumor_model.tflite")
ONNX_PATH = os.getenv("ONNX_PATH", "brain_tumor_model.onnx")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto").lower()
UNCERTAINTY_FLAG_THRESHOLD = float(os.getenv("UNCERTAINTY_FLAG_THRESHOLD", "0.05"))
PERSIST_RESULTS = os.getenv("PERSIST_RESULTS", "true").lower() not in ("false", "0", "no")


def build_model_service():
    """
    Pick an inference backend.

    'auto' prefers Keras when it is importable, because Grad-CAM++ and MC Dropout need
    it, and only falls back to ONNX otherwise. Set MODEL_BACKEND=onnx to force the
    lighter runtime and accept classification-only serving.
    """
    if MODEL_BACKEND == "onnx":
        return OnnxModelService(ONNX_PATH)

    if MODEL_BACKEND == "keras":
        return ModelService(model_path=MODEL_PATH, tflite_path=TFLITE_PATH)

    keras_service = ModelService(model_path=MODEL_PATH, tflite_path=TFLITE_PATH)
    if keras_service.is_loaded:
        return keras_service

    onnx_service = OnnxModelService(ONNX_PATH)
    return onnx_service if onnx_service.is_loaded else keras_service


class Services:
    """Container for everything built during startup."""

    model: ModelService | OnnxModelService
    quality: ImageQualityAssessor
    grader: TumorGrader
    sequence: SequenceDetector
    small_tumor: SmallTumorDetector
    pediatric: PediatricAssessor
    segmentation: SegmentationService
    ood: OODDetector


svc = Services()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup instead of at import time."""
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))

    svc.model = build_model_service()
    svc.model.set_calibration_temperature(config.calibration_temp())
    svc.quality = ImageQualityAssessor()
    svc.grader = TumorGrader()
    svc.sequence = SequenceDetector()
    svc.small_tumor = SmallTumorDetector()
    svc.pediatric = PediatricAssessor()
    svc.segmentation = SegmentationService()
    svc.ood = OODDetector()

    if PERSIST_RESULTS:
        init_db()

    log_event(
        "startup",
        model_loaded=svc.model.is_loaded,
        model_type=svc.model.model_type,
        onnx_available=ONNX_AVAILABLE,
        auth_required=auth_enabled(),
        ood_fitted=svc.ood.is_fitted,
        segmentation="unet" if svc.segmentation.is_trained_model else "gradcam_fallback",
        **config.build_info(),
    )
    yield
    log_event("shutdown")


app = FastAPI(
    title="BrainScan AI",
    description="Brain Tumor Classification API with Grad-CAM++ Explainability",
    version=config.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# Applied to every clinical endpoint.
AUTH = Depends(require_api_key)


# ── Helpers ────────────────────────────────────────────────────────────────────


def require_model() -> ModelService | OnnxModelService:
    if not svc.model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return svc.model


def require_keras() -> ModelService:
    model = require_model()
    if model.model_type != "keras":
        raise HTTPException(
            status_code=400,
            detail=(
                f"This endpoint requires the Keras backend (currently serving "
                f"'{model.model_type}'). Gradient-based explanations are unavailable "
                f"on this runtime."
            ),
        )
    return model


async def load_image(file: UploadFile) -> tuple[np.ndarray, np.ndarray, dict | None, bytes]:
    """
    Validate and decode an upload.

    Returns (model_input_240x240, raw_bgr_image, dicom_metadata, raw_bytes).
    """
    content, fmt = await read_validated(file)
    metadata = None

    if fmt == "dicom":
        if not is_dicom(content):
            raise HTTPException(status_code=400, detail="Malformed DICOM file")
        try:
            raw_img = dicom_to_image(content)
            metadata = extract_dicom_metadata(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"DICOM parsing error: {exc}") from exc
        ok, buf = cv2.imencode(".png", raw_img)
        if not ok:
            raise HTTPException(status_code=400, detail="Could not convert DICOM pixel data")
        image_bytes = buf.tobytes()
    else:
        raw_img = decode_image(content)
        image_bytes = content

    try:
        model_input = svc.model.preprocess_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return model_input, raw_img, metadata, content


def check_ood(img: np.ndarray) -> OODInfo:
    """Score an input against the training distribution."""
    logits = svc.model.predict_logits(img)
    feature = svc.model.extract_features(img)
    return OODInfo(**svc.ood.score(logits, feature).to_dict())


def _dicom_spacing(metadata: dict | None) -> tuple[float | None, float | None]:
    if not metadata:
        return None, None
    spacing = metadata.get("PixelSpacing")
    thickness = metadata.get("SliceThickness")
    try:
        spacing_mm = float(spacing[0]) if spacing else None
    except (TypeError, ValueError, IndexError):
        spacing_mm = None
    try:
        thickness_mm = float(thickness) if thickness else None
    except (TypeError, ValueError):
        thickness_mm = None
    return spacing_mm, thickness_mm


def persist_study(session: Session | None, **fields) -> str | None:
    """Store a completed analysis. Persistence failures never break inference."""
    if session is None or not PERSIST_RESULTS:
        return None
    try:
        study = Study(
            model_version=config.MODEL_VERSION,
            git_sha=config.git_sha(),
            **fields,
        )
        session.add(study)
        session.flush()
        return study.id
    except Exception as exc:  # noqa: BLE001 - persistence must not break inference
        session.rollback()
        log_event("persist_failed", level=logging.ERROR, error=str(exc))
        return None


def clinical_recommendation(
    confidence: float,
    uncertainty: float,
    ood: OODInfo | None = None,
    quality_score: float | None = None,
) -> tuple[str, bool, str]:
    """Returns (recommendation, flagged_for_review, confidence_level)."""
    if ood is not None and ood.is_out_of_distribution:
        return (
            "Input does not resemble a brain MRI from the training distribution. "
            "No diagnostic interpretation should be drawn from this result.",
            True, "rejected",
        )
    if quality_score is not None and quality_score < 50:
        return "Image quality too low for reliable interpretation. Re-acquire the scan.", True, "low"
    if uncertainty > UNCERTAINTY_FLAG_THRESHOLD:
        return "High uncertainty detected. This case requires specialist review.", True, "low"
    if confidence > 0.9:
        return "High confidence result. Consistent with AI assessment.", False, "high"
    if confidence > 0.7:
        return "Moderate confidence. Correlate with clinical findings.", False, "moderate"
    return "Low confidence — specialist review recommended before any clinical decision.", True, "low"


# ── Health & metrics ───────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API, model, and dependency status."""
    build = config.build_info()
    temp = config.calibration_temp()
    return HealthResponse(
        status="healthy" if svc.model.is_loaded else "model_not_loaded",
        model_loaded=svc.model.is_loaded,
        model_type=svc.model.model_type,
        calibration_enabled=temp != 1.0,
        calibration_temperature=temp,
        classes=list(CLASS_NAMES.values()),
        api_version=build["api_version"],
        model_version=build["model_version"],
        git_sha=build["git_sha"],
        ood_detector_fitted=svc.ood.is_fitted,
        segmentation_model="unet" if svc.segmentation.is_trained_model else "gradcam_fallback",
        database_connected=PERSIST_RESULTS,
        auth_required=auth_enabled(),
        explainability_available=svc.model.model_type == "keras",
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


# ── Prediction ─────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResult, dependencies=[AUTH, Depends(standard_rate_limit)])
async def predict(
    file: UploadFile = File(...),
    use_tta: bool = Query(False, description="Test-Time Augmentation for +1-2% accuracy"),
    use_uncertainty: bool = Query(True, description="Compute MC Dropout uncertainty"),
    use_calibration: bool = Query(True, description="Apply temperature scaling"),
    patient_id: str | None = Query(None, max_length=64),
    session: Session = Depends(get_session),
):
    """
    Classify a brain MRI scan.

    Accepts JPEG, PNG, BMP, TIFF, or DICOM. Inputs that fall outside the
    training distribution are flagged and not given a clinical interpretation.
    """
    model = require_model()
    img, _, _, content = await load_image(file)

    ood = check_ood(img) if model.model_type == "keras" else None

    if use_tta:
        probs = model.predict_with_tta(img)
    elif use_calibration:
        probs = model.predict_calibrated(img)
    else:
        probs = model.predict(img)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    uncertainty_val = None
    if use_uncertainty:
        _, std_pred = model.predict_with_uncertainty(img, n_iter=30)
        uncertainty_val = float(std_pred[pred_idx])

    _, flagged, _ = clinical_recommendation(confidence, uncertainty_val or 0.0, ood)
    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    study_id = persist_study(
        session,
        patient_id=patient_id,
        filename=file.filename,
        image_sha256=image_fingerprint(content),
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=confidence,
        uncertainty=uncertainty_val,
        probabilities=json.dumps(probabilities),
        ood_score=ood.score if ood else None,
        is_ood=bool(ood and ood.is_out_of_distribution),
        flagged_for_review=flagged,
    )

    metrics.observe_prediction(CLASS_NAMES[pred_idx], flagged, bool(ood and ood.is_out_of_distribution))
    build = config.build_info()

    return PredictionResult(
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=confidence,
        probabilities=probabilities,
        uncertainty=uncertainty_val,
        calibrated_confidence=confidence if use_calibration else None,
        flagged_for_review=flagged,
        out_of_distribution=ood,
        study_id=study_id,
        model_version=build["model_version"],
        git_sha=build["git_sha"],
    )


@app.post(
    "/predict/batch",
    response_model=list[BatchPredictionResult],
    dependencies=[AUTH, Depends(batch_rate_limit)],
)
async def predict_batch(
    files: list[UploadFile] = File(...),
    use_uncertainty: bool = Query(True),
):
    """
    Batch prediction — upload multiple MRI scans at once.

    A failure on one file does not abort the batch; that entry reports an error.
    """
    model = require_model()

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Max 100 files per batch")

    build = config.build_info()
    results: list[BatchPredictionResult] = []

    for file in files:
        try:
            img, _, _, _ = await load_image(file)
            probs = model.predict(img)
            pred_idx = int(np.argmax(probs))

            uncertainty_val = None
            if use_uncertainty:
                _, std_pred = model.predict_with_uncertainty(img, n_iter=20)
                uncertainty_val = float(std_pred[pred_idx])

            flagged = (
                (uncertainty_val or 0.0) > UNCERTAINTY_FLAG_THRESHOLD
                or float(probs[pred_idx]) < 0.7
            )
            results.append(BatchPredictionResult(
                filename=file.filename or "unknown",
                result=PredictionResult(
                    predicted_class=CLASS_NAMES[pred_idx],
                    confidence=float(probs[pred_idx]),
                    probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
                    uncertainty=uncertainty_val,
                    flagged_for_review=flagged,
                    model_version=build["model_version"],
                    git_sha=build["git_sha"],
                ),
            ))
        except HTTPException as exc:
            log_event("batch_item_failed", level=logging.WARNING,
                      filename=file.filename, detail=str(exc.detail))
            results.append(BatchPredictionResult(
                filename=file.filename or "unknown",
                result=PredictionResult(
                    predicted_class="Error",
                    confidence=0.0,
                    probabilities={},
                    flagged_for_review=True,
                ),
            ))

    return results


# ── Explainability ─────────────────────────────────────────────────────────────


@app.post("/explain/gradcam", response_model=GradCAMResult, dependencies=[AUTH, Depends(standard_rate_limit)])
async def explain_gradcam(file: UploadFile = File(...)):
    """Grad-CAM++ heatmap overlay, returned as base64 PNGs."""
    model = require_keras()
    img, _, _, _ = await load_image(file)
    heatmap, overlay, pred_idx = model.gradcam_plus_plus(img)
    probs = model.predict(img)

    return GradCAMResult(
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        heatmap_base64=model.encode_image(heatmap),
        overlay_base64=model.encode_image(overlay),
    )


@app.post("/explain/shap", response_model=SHAPResult, dependencies=[AUTH, Depends(heavy_rate_limit)])
async def explain_shap(file: UploadFile = File(...)):
    """SHAP pixel attribution — slower than Grad-CAM++ but theoretically grounded."""
    model = require_keras()

    try:
        from api.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(model.model)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="SHAP not installed: pip install shap") from exc

    img, _, _, _ = await load_image(file)
    probs = model.predict(img)
    pred_idx = int(np.argmax(probs))

    return SHAPResult(
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        shap_image_base64=explainer.explain(img),
    )


# ── Segmentation & volumetry ───────────────────────────────────────────────────


@app.post("/segment", response_model=SegmentationResult, dependencies=[AUTH, Depends(heavy_rate_limit)])
async def segment_tumor(
    file: UploadFile = File(...),
    patient_id: str | None = Query(None, max_length=64),
    pixel_spacing: float | None = Query(None, gt=0, description="mm per pixel; from DICOM when omitted"),
    slice_thickness: float | None = Query(None, gt=0, description="mm; from DICOM when omitted"),
    threshold: float = Query(0.5, ge=0.05, le=0.95),
    session: Session = Depends(get_session),
):
    """
    Delineate the tumor and measure it.

    Returns a mask, a contoured overlay, and physical measurements. Storing the
    volume here is what makes /patients/{id}/timeline growth rates meaningful.
    """
    model = require_keras()
    img, raw_img, metadata, content = await load_image(file)

    dicom_spacing, dicom_thickness = _dicom_spacing(metadata)
    spacing = pixel_spacing or dicom_spacing
    thickness = slice_thickness or dicom_thickness

    heatmap = None
    if not svc.segmentation.is_trained_model:
        heatmap, _, _ = model.gradcam_plus_plus(img)

    result = svc.segmentation.segment(
        img=img,
        heatmap_rgb=heatmap,
        pixel_spacing_mm=spacing,
        slice_thickness_mm=thickness,
        threshold=threshold,
        original_shape=raw_img.shape[:2],
    )

    probs = model.predict(img)
    pred_idx = int(np.argmax(probs))
    result["predicted_class"] = CLASS_NAMES[pred_idx]
    result["confidence"] = float(probs[pred_idx])

    result["study_id"] = persist_study(
        session,
        patient_id=patient_id,
        filename=file.filename,
        image_sha256=image_fingerprint(content),
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities=json.dumps({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}),
        tumor_volume_mm3=result["measurements"]["volume_mm3"],
        tumor_area_px=result["measurements"]["area_pixels"],
        flagged_for_review=False,
    )

    return SegmentationResult(**result)


# ── Full analysis ──────────────────────────────────────────────────────────────


@app.post("/analyze", dependencies=[AUTH, Depends(heavy_rate_limit)])
async def full_analysis(
    file: UploadFile = File(...),
    patient_id: str | None = Query(None, max_length=64),
    session: Session = Depends(get_session),
):
    """
    Complete analysis for the doctor-facing UI.

    Prediction + uncertainty + OOD check + Grad-CAM++ + clinical recommendation.
    """
    model = require_model()
    img, _, metadata, content = await load_image(file)

    ood = check_ood(img) if model.model_type == "keras" else None

    mean_pred, std_pred = model.predict_with_uncertainty(img, n_iter=30)
    pred_idx = int(np.argmax(mean_pred))
    confidence = float(mean_pred[pred_idx])
    uncertainty = float(std_pred[pred_idx])

    heatmap_b64, overlay_b64 = "", ""
    if model.model_type == "keras":
        heatmap, overlay, _ = model.gradcam_plus_plus(img)
        heatmap_b64 = model.encode_image(heatmap)
        overlay_b64 = model.encode_image(overlay)

    recommendation, flagged, level = clinical_recommendation(confidence, uncertainty, ood)
    probabilities = {CLASS_NAMES[i]: round(float(mean_pred[i]) * 100, 1) for i in range(len(CLASS_NAMES))}

    study_id = persist_study(
        session,
        patient_id=patient_id,
        filename=file.filename,
        image_sha256=image_fingerprint(content),
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=confidence,
        uncertainty=uncertainty,
        probabilities=json.dumps(probabilities),
        ood_score=ood.score if ood else None,
        is_ood=bool(ood and ood.is_out_of_distribution),
        flagged_for_review=flagged,
        gradcam_overlay=overlay_b64 or None,
    )

    metrics.observe_prediction(CLASS_NAMES[pred_idx], flagged, bool(ood and ood.is_out_of_distribution))

    return {
        "study_id": study_id,
        "prediction": {
            "class": CLASS_NAMES[pred_idx],
            "confidence": round(confidence * 100, 1),
            "uncertainty": round(uncertainty * 100, 1),
            "all_probabilities": probabilities,
        },
        "explainability": {
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
        },
        "out_of_distribution": ood.model_dump() if ood else None,
        "clinical": {
            "recommendation": recommendation,
            "flagged_for_review": flagged,
            "confidence_level": level,
        },
        "dicom_metadata": metadata,
        "build": config.build_info(),
    }


@app.post("/analyze/comprehensive", dependencies=[AUTH, Depends(heavy_rate_limit)])
async def comprehensive_analysis(
    file: UploadFile = File(...),
    patient_age: int | None = Query(None, ge=0, le=120),
    patient_sex: str | None = Query(None, pattern="^[MFmf]$"),
    patient_id: str | None = Query(None, max_length=64),
    include_segmentation: bool = Query(True),
    session: Session = Depends(get_session),
):
    """
    Everything in one call: quality, sequence, OOD, classification, grading,
    segmentation volumetry, Grad-CAM++, pediatric adjustment, recommendation.
    """
    model = require_model()
    img, raw_img, metadata, content = await load_image(file)

    quality = svc.quality.assess(raw_img)
    sequence = svc.sequence.detect(raw_img, dicom_metadata=metadata)
    ood = check_ood(img) if model.model_type == "keras" else None

    mean_pred, std_pred = model.predict_with_uncertainty(img, n_iter=30)
    pred_idx = int(np.argmax(mean_pred))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(mean_pred[pred_idx])
    uncertainty = float(std_pred[pred_idx])
    probabilities = {CLASS_NAMES[i]: float(mean_pred[i]) for i in range(len(CLASS_NAMES))}

    heatmap_b64, overlay_b64, heatmap = "", "", None
    if model.model_type == "keras":
        heatmap, overlay, _ = model.gradcam_plus_plus(img)
        heatmap_b64 = model.encode_image(heatmap)
        overlay_b64 = model.encode_image(overlay)

    grading = svc.grader.estimate_grade(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        image=img,
        heatmap=heatmap,
    )

    segmentation = None
    if include_segmentation and model.model_type == "keras":
        spacing, thickness = _dicom_spacing(metadata)
        segmentation = svc.segmentation.segment(
            img=img, heatmap_rgb=heatmap,
            pixel_spacing_mm=spacing, slice_thickness_mm=thickness,
            original_shape=raw_img.shape[:2],
        )

    pediatric = None
    if patient_age is not None and patient_age < 18:
        pediatric = svc.pediatric.assess(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            patient_age=patient_age,
            patient_sex=patient_sex,
        )

    recommendation, flagged, level = clinical_recommendation(
        confidence, uncertainty, ood, quality.get("overall_score"),
    )

    study_id = persist_study(
        session,
        patient_id=patient_id,
        filename=file.filename,
        image_sha256=image_fingerprint(content),
        predicted_class=predicted_class,
        confidence=confidence,
        uncertainty=uncertainty,
        probabilities=json.dumps({k: round(v * 100, 1) for k, v in probabilities.items()}),
        quality_score=quality.get("overall_score"),
        sequence_type=str(sequence.get("detected_sequence")) if sequence else None,
        who_grade=str(grading.get("estimated_grade")) if grading else None,
        tumor_volume_mm3=segmentation["measurements"]["volume_mm3"] if segmentation else None,
        tumor_area_px=segmentation["measurements"]["area_pixels"] if segmentation else None,
        ood_score=ood.score if ood else None,
        is_ood=bool(ood and ood.is_out_of_distribution),
        flagged_for_review=flagged,
        gradcam_overlay=overlay_b64 or None,
    )

    metrics.observe_prediction(predicted_class, flagged, bool(ood and ood.is_out_of_distribution))

    return {
        "study_id": study_id,
        "image_quality": quality,
        "sequence": sequence,
        "out_of_distribution": ood.model_dump() if ood else None,
        "prediction": {
            "class": predicted_class,
            "confidence": round(confidence * 100, 1),
            "uncertainty": round(uncertainty * 100, 1),
            "all_probabilities": {k: round(v * 100, 1) for k, v in probabilities.items()},
        },
        "grading": grading,
        "segmentation": segmentation,
        "explainability": {
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
        },
        "pediatric": pediatric,
        "clinical": {
            "recommendation": recommendation,
            "flagged_for_review": flagged,
            "confidence_level": level,
        },
        "dicom_metadata": metadata,
        "build": config.build_info(),
    }


# ── Assessment modules ─────────────────────────────────────────────────────────


@app.post("/assess/quality", dependencies=[AUTH, Depends(standard_rate_limit)])
async def assess_image_quality(file: UploadFile = File(...)):
    """Pre-inference quality check: resolution, blur, noise, compression, coverage."""
    content, fmt = await read_validated(file)
    raw_img = dicom_to_image(content) if fmt == "dicom" else decode_image(content)
    return svc.quality.assess(raw_img)


@app.post("/assess/sequence", dependencies=[AUTH, Depends(standard_rate_limit)])
async def detect_sequence(file: UploadFile = File(...)):
    """Auto-detect MRI sequence (T1/T1CE/T2/FLAIR/DWI)."""
    content, fmt = await read_validated(file)

    metadata = None
    if fmt == "dicom":
        try:
            metadata = extract_dicom_metadata(content)
            raw_img = dicom_to_image(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"DICOM error: {exc}") from exc
    else:
        raw_img = decode_image(content)

    return svc.sequence.detect(raw_img, dicom_metadata=metadata)


@app.post("/assess/grade", dependencies=[AUTH, Depends(heavy_rate_limit)])
async def estimate_tumor_grade(file: UploadFile = File(...)):
    """
    Estimate WHO tumor grade (I-IV).

    A computational estimate only — definitive grading requires histopathology.
    """
    model = require_model()
    img, _, _, _ = await load_image(file)

    probs = model.predict(img)
    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    heatmap = None
    if model.model_type == "keras":
        heatmap, _, _ = model.gradcam_plus_plus(img)

    result = svc.grader.estimate_grade(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        image=img,
        heatmap=heatmap,
    )
    result["prediction"] = {"class": predicted_class, "confidence": round(confidence * 100, 1)}
    return result


@app.post("/assess/pediatric", dependencies=[AUTH, Depends(heavy_rate_limit)])
async def pediatric_assessment(
    file: UploadFile = File(...),
    patient_age: int | None = Query(None, ge=0, le=120, description="Patient age in years"),
    patient_sex: str | None = Query(None, pattern="^[MFmf]$"),
):
    """Age-adjusted assessment using pediatric tumor priors."""
    model = require_model()
    img, _, metadata, _ = await load_image(file)

    if patient_age is None and metadata:
        dicom_age = metadata.get("PatientAge")
        if dicom_age:
            try:
                patient_age = int(str(dicom_age).upper().rstrip("Y"))
            except (ValueError, TypeError):
                patient_age = None

    mean_pred, std_pred = model.predict_with_uncertainty(img, n_iter=30)
    pred_idx = int(np.argmax(mean_pred))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(mean_pred[pred_idx])

    pedi_result = svc.pediatric.assess(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(mean_pred[i]) for i in range(len(CLASS_NAMES))},
        patient_age=patient_age,
        patient_sex=patient_sex,
    )

    return {
        "prediction": {
            "class": predicted_class,
            "confidence": round(confidence * 100, 1),
            "uncertainty": round(float(std_pred[pred_idx]) * 100, 1),
        },
        "pediatric_assessment": pedi_result,
    }


@app.post("/detect/small-tumors", dependencies=[AUTH, Depends(heavy_rate_limit)])
async def detect_small_tumors(
    file: UploadFile = File(...),
    sensitivity: str = Query("high", pattern="^(low|medium|high)$"),
    pixel_spacing: float | None = Query(None, gt=0, description="mm per pixel (from DICOM)"),
):
    """
    Sliding-window patch analysis for sub-5mm lesions that whole-image
    classification tends to miss.
    """
    model = require_keras()
    _, raw_img, metadata, _ = await load_image(file)

    if pixel_spacing is None:
        pixel_spacing, _ = _dicom_spacing(metadata)

    return svc.small_tumor.detect(
        image=raw_img,
        model=model.model,
        pixel_spacing=pixel_spacing,
        sensitivity=sensitivity,
    )


# ── Studies, feedback, timelines, reports ──────────────────────────────────────


def _to_summary(study: Study) -> StudySummary:
    summary = StudySummary.model_validate(study)
    summary.confirmed_class = study.feedback.corrected_class if study.feedback else None
    return summary


@app.get("/studies", response_model=list[StudySummary], dependencies=[AUTH, Depends(standard_rate_limit)])
async def list_studies(
    patient_id: str | None = Query(None, max_length=64),
    flagged_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """Browse stored analyses — the backing store for the review worklist."""
    stmt = select(Study).order_by(Study.created_at.desc())
    if patient_id:
        stmt = stmt.where(Study.patient_id == patient_id)
    if flagged_only:
        stmt = stmt.where(Study.flagged_for_review.is_(True))

    studies = session.execute(stmt.limit(limit).offset(offset)).scalars().all()
    return [_to_summary(s) for s in studies]


@app.get("/studies/{study_id}", response_model=StudySummary, dependencies=[AUTH, Depends(standard_rate_limit)])
async def get_study(study_id: str, session: Session = Depends(get_session)):
    study = session.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return _to_summary(study)


@app.post(
    "/studies/{study_id}/feedback",
    response_model=FeedbackResponse,
    dependencies=[AUTH, Depends(standard_rate_limit)],
)
async def submit_feedback(
    study_id: str,
    payload: FeedbackRequest,
    session: Session = Depends(get_session),
):
    """
    Record the radiologist's verdict.

    Disagreements are the hard examples worth retraining on.
    """
    study = session.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")

    valid_classes = set(CLASS_NAMES.values())
    if payload.corrected_class not in valid_classes:
        raise HTTPException(
            status_code=400,
            detail=f"corrected_class must be one of: {', '.join(sorted(valid_classes))}",
        )

    agrees = payload.corrected_class == study.predicted_class
    record = study.feedback
    if record is not None:
        record.corrected_class = payload.corrected_class
        record.reviewer = payload.reviewer
        record.notes = payload.notes
        record.agrees_with_ai = agrees
    else:
        record = Feedback(
            study_id=study_id,
            corrected_class=payload.corrected_class,
            reviewer=payload.reviewer,
            notes=payload.notes,
            agrees_with_ai=agrees,
        )
        session.add(record)

    session.flush()
    log_event("feedback_recorded", study_id=study_id, agrees_with_ai=agrees,
              predicted=study.predicted_class, corrected=payload.corrected_class)

    return FeedbackResponse(
        study_id=study_id,
        corrected_class=record.corrected_class,
        agrees_with_ai=agrees,
        reviewer=record.reviewer,
        recorded_at=record.created_at or datetime.now(UTC),
    )


@app.get("/patients/{patient_id}/timeline", dependencies=[AUTH, Depends(standard_rate_limit)])
async def patient_timeline(patient_id: str, session: Session = Depends(get_session)):
    """
    Longitudinal progression analysis across every stored study for a patient.

    Growth rates come from segmentation volumes, so at least two segmented
    scans are needed for a trend.
    """
    result = analyze_patient(session, patient_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="No studies found for this patient")
    return result


@app.get("/studies/{study_id}/report", dependencies=[AUTH, Depends(standard_rate_limit)])
async def download_report(
    study_id: str,
    include_timeline: bool = Query(True),
    session: Session = Depends(get_session),
):
    """Render the study as a PDF report suitable for attaching to a record."""
    study = session.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")

    try:
        from api.reports import build_study_report
    except ImportError as exc:
        raise HTTPException(
            status_code=501, detail="PDF export unavailable: pip install reportlab",
        ) from exc

    timeline = None
    if include_timeline and study.patient_id:
        from api.longitudinal_service import fetch_timeline
        timeline = fetch_timeline(session, study.patient_id)

    pdf_bytes = build_study_report(study, config.build_info(), timeline)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="brainscan-report-{study_id[:8]}.pdf"'},
    )
