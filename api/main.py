"""
FastAPI Backend — Upgrade #10.

Serves the brain tumor classification model via REST API.
Doctors upload MRI scans → get prediction + Grad-CAM++ + uncertainty.

Includes:
- Single & batch prediction (#14)
- DICOM support (#13)
- Grad-CAM++ visualization
- SHAP explainability (#6)
- Calibrated confidence (#9)
- MC Dropout uncertainty
- TTA accuracy boost
"""

import io
import os
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    PredictionResult, BatchPredictionResult,
    GradCAMResult, SHAPResult, HealthResponse,
)
from api.model_service import ModelService, CLASS_NAMES
from api.dicom_handler import is_dicom, dicom_to_image, extract_dicom_metadata
from api.calibration import TemperatureCalibrator
from api.image_quality import ImageQualityAssessor
from api.tumor_grading import TumorGrader
from api.sequence_detector import SequenceDetector
from api.small_tumor_detector import SmallTumorDetector
from api.pediatric_support import PediatricAssessor

import numpy as np
import cv2

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BrainScan AI",
    description="Brain Tumor Classification API with Grad-CAM++ Explainability",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ──────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model_best.keras")
TFLITE_PATH = os.getenv("TFLITE_PATH", "brain_tumor_model.tflite")
CALIBRATION_TEMP = float(os.getenv("CALIBRATION_TEMP", "1.0"))

model_svc = ModelService(model_path=MODEL_PATH, tflite_path=TFLITE_PATH)
if CALIBRATION_TEMP != 1.0:
    model_svc.set_calibration_temperature(CALIBRATION_TEMP)

# ── New Feature Services ───────────────────────────────────────────────────────
quality_assessor = ImageQualityAssessor()
tumor_grader = TumorGrader()
sequence_detector = SequenceDetector()
small_tumor_detector = SmallTumorDetector()
pediatric_assessor = PediatricAssessor()

# ── Helpers ────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".dicom"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def validate_upload(file: UploadFile) -> None:
    """Validate file type and size."""
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
            )


async def read_and_preprocess(file: UploadFile) -> "tuple[np.ndarray, Optional[dict]]":
    """Read upload, handle DICOM or standard image, return preprocessed array."""
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    metadata = None
    if is_dicom(content):
        try:
            raw_img = dicom_to_image(content)
            metadata = extract_dicom_metadata(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DICOM parsing error: {str(e)}")
        # Encode as PNG bytes for model_service preprocessing
        _, buf = cv2.imencode(".png", raw_img)
        content = buf.tobytes()

    img = model_svc.preprocess_image(content)
    return img, metadata


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="healthy" if model_svc.is_loaded else "model_not_loaded",
        model_loaded=model_svc.is_loaded,
        model_type=model_svc.model_type,
        calibration_enabled=CALIBRATION_TEMP != 1.0,
        classes=list(CLASS_NAMES.values()),
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    use_tta: bool = Query(False, description="Use Test-Time Augmentation for +1-2% accuracy"),
    use_uncertainty: bool = Query(True, description="Compute MC Dropout uncertainty"),
    use_calibration: bool = Query(True, description="Apply temperature scaling"),
):
    """
    Classify a brain MRI scan.

    Accepts JPG, PNG, BMP, TIFF, or DICOM files.
    Returns tumor class, confidence, uncertainty, and review flag.
    """
    validate_upload(file)

    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img, _ = await read_and_preprocess(file)

    # Prediction
    if use_tta:
        probs = model_svc.predict_with_tta(img)
    elif use_calibration and CALIBRATION_TEMP != 1.0:
        probs = model_svc.predict_calibrated(img)
    else:
        probs = model_svc.predict(img)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # Uncertainty
    uncertainty_val = None
    flagged = False
    if use_uncertainty:
        mean_pred, std_pred = model_svc.predict_with_uncertainty(img, n_iter=30)
        uncertainty_val = float(std_pred[pred_idx])
        flagged = uncertainty_val > 0.05  # >5% uncertainty = flag for review

    return PredictionResult(
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(4)},
        uncertainty=uncertainty_val,
        calibrated_confidence=float(probs[pred_idx]) if use_calibration else None,
        flagged_for_review=flagged,
    )


@app.post("/predict/batch", response_model=list[BatchPredictionResult])
async def predict_batch(
    files: list[UploadFile] = File(...),
    use_uncertainty: bool = Query(True),
):
    """
    Batch prediction — upload multiple MRI scans at once (#14).

    Radiologists often need to review dozens of scans.
    Returns results for each file with priority flagging.
    """
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Max 100 files per batch")

    results = []
    for file in files:
        try:
            validate_upload(file)
            img, _ = await read_and_preprocess(file)
            probs = model_svc.predict(img)
            pred_idx = int(np.argmax(probs))

            uncertainty_val = None
            flagged = False
            if use_uncertainty:
                _, std_pred = model_svc.predict_with_uncertainty(img, n_iter=20)
                uncertainty_val = float(std_pred[pred_idx])
                flagged = uncertainty_val > 0.05

            results.append(BatchPredictionResult(
                filename=file.filename or "unknown",
                result=PredictionResult(
                    predicted_class=CLASS_NAMES[pred_idx],
                    confidence=float(probs[pred_idx]),
                    probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(4)},
                    uncertainty=uncertainty_val,
                    flagged_for_review=flagged,
                )
            ))
        except HTTPException:
            raise
        except Exception as e:
            results.append(BatchPredictionResult(
                filename=file.filename or "unknown",
                result=PredictionResult(
                    predicted_class="Error",
                    confidence=0.0,
                    probabilities={},
                    flagged_for_review=True,
                )
            ))

    return results


@app.post("/explain/gradcam", response_model=GradCAMResult)
async def explain_gradcam(file: UploadFile = File(...)):
    """
    Generate Grad-CAM++ heatmap overlay for an MRI scan.

    Returns the heatmap and overlay as base64-encoded PNG images.
    Requires Keras model (not TFLite).
    """
    validate_upload(file)
    if model_svc.model_type != "keras":
        raise HTTPException(status_code=400, detail="Grad-CAM++ requires Keras model")

    img, _ = await read_and_preprocess(file)
    heatmap, overlay, pred_idx = model_svc.gradcam_plus_plus(img)
    probs = model_svc.predict(img)

    return GradCAMResult(
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        heatmap_base64=model_svc.encode_image(heatmap),
        overlay_base64=model_svc.encode_image(overlay),
    )


@app.post("/explain/shap", response_model=SHAPResult)
async def explain_shap(file: UploadFile = File(...)):
    """
    Generate SHAP feature attribution (#6).

    Shows which pixel regions support or oppose the prediction.
    Slower than Grad-CAM++ but theoretically grounded.
    """
    if model_svc.model_type != "keras":
        raise HTTPException(status_code=400, detail="SHAP requires Keras model")

    try:
        from api.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(model_svc.model)
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP not installed: pip install shap")

    img, _ = await read_and_preprocess(file)
    probs = model_svc.predict(img)
    pred_idx = int(np.argmax(probs))

    shap_b64 = explainer.explain(img)

    return SHAPResult(
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        shap_image_base64=shap_b64,
    )


@app.post("/analyze")
async def full_analysis(file: UploadFile = File(...)):
    """
    Complete analysis endpoint for the doctor-facing UI.

    Returns prediction + confidence + uncertainty + Grad-CAM++ + clinical recommendation.
    Single endpoint for the frontend to call.
    """
    validate_upload(file)
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    await file.seek(0)  # Reset for re-reading

    # Handle DICOM
    metadata = None
    if is_dicom(content):
        try:
            raw_img = dicom_to_image(content)
            metadata = extract_dicom_metadata(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DICOM error: {str(e)}")
        _, buf = cv2.imencode(".png", raw_img)
        content = buf.tobytes()

    img = model_svc.preprocess_image(content)

    # Prediction with uncertainty
    mean_pred, std_pred = model_svc.predict_with_uncertainty(img, n_iter=30)
    pred_idx = int(np.argmax(mean_pred))
    confidence = float(mean_pred[pred_idx])
    uncertainty = float(std_pred[pred_idx])

    # Grad-CAM++
    heatmap_b64, overlay_b64 = "", ""
    if model_svc.model_type == "keras":
        heatmap, overlay, _ = model_svc.gradcam_plus_plus(img)
        heatmap_b64 = model_svc.encode_image(heatmap)
        overlay_b64 = model_svc.encode_image(overlay)

    # Clinical recommendation
    if confidence > 0.9 and uncertainty < 0.03:
        recommendation = "High confidence result. Consistent with AI assessment."
    elif confidence > 0.7:
        recommendation = "Moderate confidence. Consider correlation with clinical findings."
    else:
        recommendation = "Low confidence — specialist review recommended before clinical decision."

    if uncertainty > 0.05:
        recommendation = "⚠️ High uncertainty detected. This case requires specialist review."

    return {
        "prediction": {
            "class": CLASS_NAMES[pred_idx],
            "confidence": round(confidence * 100, 1),
            "uncertainty": round(uncertainty * 100, 1),
            "all_probabilities": {CLASS_NAMES[i]: round(float(mean_pred[i]) * 100, 1) for i in range(4)},
        },
        "explainability": {
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
        },
        "clinical": {
            "recommendation": recommendation,
            "flagged_for_review": uncertainty > 0.05 or confidence < 0.7,
            "confidence_level": "high" if confidence > 0.9 else "moderate" if confidence > 0.7 else "low",
        },
        "dicom_metadata": metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NEW ENDPOINTS: Image Quality, Tumor Grading, Sequence Detection,
#                Small Tumor Detection, Pediatric Support
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/assess/quality")
async def assess_image_quality(file: UploadFile = File(...)):
    """
    Pre-inference image quality assessment.

    Checks resolution, blur, noise, compression artifacts, and brain coverage.
    Returns a quality score (0-100) and specific issues with recommendations.
    """
    validate_upload(file)
    content = await file.read()

    # Decode image
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = quality_assessor.assess(img)
    return result


@app.post("/assess/sequence")
async def detect_sequence(file: UploadFile = File(...)):
    """
    Auto-detect MRI sequence type (T1, T2, FLAIR, DWI).

    Returns the detected sequence, confidence, and preprocessing advice.
    If a DICOM file is uploaded, uses metadata for accurate detection.
    """
    validate_upload(file)
    content = await file.read()

    metadata = None
    if is_dicom(content):
        try:
            metadata = extract_dicom_metadata(content)
            raw_img = dicom_to_image(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DICOM error: {str(e)}")
    else:
        arr = np.frombuffer(content, np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if raw_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = sequence_detector.detect(raw_img, dicom_metadata=metadata)
    return result


@app.post("/assess/grade")
async def estimate_tumor_grade(file: UploadFile = File(...)):
    """
    Estimate WHO tumor grade (I-IV) from the classification result.

    Combines model prediction with imaging features (heterogeneity,
    size, enhancement pattern) to estimate likely grade.

    NOTE: This is a computational estimate — definitive grading requires
    histopathological analysis.
    """
    validate_upload(file)
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img, _ = await read_and_preprocess(file)

    # Get prediction
    probs = model_svc.predict(img)
    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Get Grad-CAM for spatial analysis
    heatmap = None
    if model_svc.model_type == "keras":
        heatmap, _, _ = model_svc.gradcam_plus_plus(img)

    # Grade estimation
    result = tumor_grader.estimate_grade(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(4)},
        image=img,
        heatmap=heatmap,
    )

    result["prediction"] = {
        "class": predicted_class,
        "confidence": round(confidence * 100, 1),
    }

    return result


@app.post("/assess/pediatric")
async def pediatric_assessment(
    file: UploadFile = File(...),
    patient_age: Optional[int] = Query(None, description="Patient age in years"),
    patient_sex: Optional[str] = Query(None, description="Patient sex (M/F)"),
):
    """
    Pediatric-adjusted brain tumor assessment.

    Provides age-appropriate confidence adjustments, differential diagnoses,
    and clinical recommendations specific to pediatric neuro-oncology.

    Addresses the limitation that the model was trained primarily on adult data.
    """
    validate_upload(file)
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if patient_age is not None and patient_age < 0:
        raise HTTPException(status_code=400, detail="Age must be non-negative")

    img, metadata = await read_and_preprocess(file)

    # Try to get age from DICOM if not provided
    if patient_age is None and metadata:
        dicom_age = metadata.get("PatientAge")
        if dicom_age:
            try:
                patient_age = int(str(dicom_age).replace("Y", ""))
            except (ValueError, TypeError):
                pass

    # Get prediction
    mean_pred, std_pred = model_svc.predict_with_uncertainty(img, n_iter=30)
    pred_idx = int(np.argmax(mean_pred))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(mean_pred[pred_idx])

    # Pediatric assessment
    pedi_result = pediatric_assessor.assess(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(mean_pred[i]) for i in range(4)},
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


@app.post("/detect/small-tumors")
async def detect_small_tumors(
    file: UploadFile = File(...),
    sensitivity: str = Query("high", description="Detection sensitivity: low, medium, high"),
    pixel_spacing: Optional[float] = Query(None, description="Pixel spacing in mm (from DICOM)"),
):
    """
    Small tumor detection using multi-scale patch analysis.

    Detects tumors smaller than ~5mm that standard classification may miss.
    Uses sliding window at multiple resolutions for comprehensive scanning.

    Best results with high-resolution DICOM input.
    """
    validate_upload(file)
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if model_svc.model_type != "keras":
        raise HTTPException(status_code=400, detail="Small tumor detection requires Keras model")

    if sensitivity not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="Sensitivity must be: low, medium, high")

    content = await file.read()

    # Get high-res image (don't resize to 240x240 yet)
    metadata = None
    if is_dicom(content):
        try:
            raw_img = dicom_to_image(content)
            metadata = extract_dicom_metadata(content)
            if pixel_spacing is None and metadata:
                ps = metadata.get("PixelSpacing")
                if ps and len(ps) > 0:
                    pixel_spacing = float(ps[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DICOM error: {str(e)}")
    else:
        arr = np.frombuffer(content, np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if raw_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = small_tumor_detector.detect(
        image=raw_img,
        model=model_svc.model,
        pixel_spacing=pixel_spacing,
        sensitivity=sensitivity,
    )

    return result


@app.post("/analyze/comprehensive")
async def comprehensive_analysis(
    file: UploadFile = File(...),
    patient_age: Optional[int] = Query(None, description="Patient age in years"),
    patient_sex: Optional[str] = Query(None, description="Patient sex (M/F)"),
):
    """
    Full comprehensive analysis combining ALL capabilities:
    - Image quality assessment
    - MRI sequence detection
    - Tumor classification with uncertainty
    - WHO grade estimation
    - Grad-CAM++ explainability
    - Pediatric adjustments (if applicable)
    - Clinical recommendations

    This is the ultimate endpoint for complete clinical decision support.
    """
    validate_upload(file)
    if not model_svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()

    # Decode raw image for quality/sequence checks
    metadata = None
    if is_dicom(content):
        try:
            raw_img = dicom_to_image(content)
            metadata = extract_dicom_metadata(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DICOM error: {str(e)}")
        _, buf = cv2.imencode(".png", raw_img)
        img_bytes = buf.tobytes()
    else:
        arr = np.frombuffer(content, np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        img_bytes = content

    # 1. Image Quality
    quality = quality_assessor.assess(raw_img)

    # 2. Sequence Detection
    sequence = sequence_detector.detect(raw_img, dicom_metadata=metadata)

    # 3. Preprocess and predict
    img = model_svc.preprocess_image(img_bytes)
    mean_pred, std_pred = model_svc.predict_with_uncertainty(img, n_iter=30)
    pred_idx = int(np.argmax(mean_pred))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(mean_pred[pred_idx])
    uncertainty = float(std_pred[pred_idx])

    # 4. Grad-CAM++
    heatmap_b64, overlay_b64 = "", ""
    heatmap = None
    if model_svc.model_type == "keras":
        heatmap, overlay, _ = model_svc.gradcam_plus_plus(img)
        heatmap_b64 = model_svc.encode_image(heatmap)
        overlay_b64 = model_svc.encode_image(overlay)

    # 5. Tumor Grading
    grading = tumor_grader.estimate_grade(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(mean_pred[i]) for i in range(4)},
        image=img,
        heatmap=heatmap,
    )

    # 6. Pediatric Assessment
    pedi = None
    if patient_age is not None and patient_age < 18:
        pedi = pediatric_assessor.assess(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities={CLASS_NAMES[i]: float(mean_pred[i]) for i in range(4)},
            patient_age=patient_age,
            patient_sex=patient_sex,
        )

    # 7. Clinical Recommendation (enhanced)
    if quality["overall_score"] < 50:
        recommendation = "⚠️ Image quality too low for reliable diagnosis. Re-acquire scan."
    elif uncertainty > 0.05:
        recommendation = "⚠️ High uncertainty — specialist review required."
    elif confidence > 0.9:
        recommendation = "High confidence result. Consistent with AI assessment."
    elif confidence > 0.7:
        recommendation = "Moderate confidence. Correlate with clinical findings."
    else:
        recommendation = "Low confidence — specialist review recommended."

    return {
        "image_quality": quality,
        "sequence": sequence,
        "prediction": {
            "class": predicted_class,
            "confidence": round(confidence * 100, 1),
            "uncertainty": round(uncertainty * 100, 1),
            "all_probabilities": {CLASS_NAMES[i]: round(float(mean_pred[i]) * 100, 1) for i in range(4)},
        },
        "grading": grading,
        "explainability": {
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
        },
        "pediatric": pedi,
        "clinical": {
            "recommendation": recommendation,
            "flagged_for_review": uncertainty > 0.05 or confidence < 0.7 or quality["overall_score"] < 50,
            "confidence_level": "high" if confidence > 0.9 else "moderate" if confidence > 0.7 else "low",
        },
        "dicom_metadata": metadata,
    }
