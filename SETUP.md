# Brain Tumor Classification with Grad-CAM++ Explainability

## Complete Setup & Run Guide

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Environment Setup](#2-clone--environment-setup)
3. [Download Datasets](#3-download-datasets)
4. [Run the Training Pipeline](#4-run-the-training-pipeline)
5. [Run the FastAPI Backend](#5-run-the-fastapi-backend)
6. [Run the React Website](#6-run-the-react-website)
7. [Run the Multi-Dataset Test Suite](#7-run-the-multi-dataset-test-suite)
8. [Run with Docker](#8-run-with-docker)
9. [Run Individual Upgrade Modules](#9-run-individual-upgrade-modules)
10. [Project Structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Requirement     | Version   |
|-----------------|-----------|
| Python          | 3.10+     |
| Node.js         | 18+       |
| pip             | 23+       |
| Docker (optional) | 24+    |
| GPU (optional)  | CUDA 11.8+ for GPU acceleration |

Verify:
```bash
python --version
node --version
pip --version
```

---

## 2. Clone & Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Brain-Tumor-Classification-with-Grad-CAM-explainability.

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install Python dependencies (training + API)
pip install tensorflow opencv-python scikit-learn imutils seaborn tqdm Pillow scikit-image matplotlib pandas gradio
pip install -r requirements-api.txt

# Install website dependencies
cd website
npm install
cd ..
```

---

## 3. Download Datasets

### Option A: Kaggle API (recommended)

> **Never commit `kaggle.json`.** Download your own token from
> <https://www.kaggle.com/settings> → *Create New Token* and place it directly in
> `~/.kaggle/`. The file is listed in `.gitignore`, and CI fails the build if it is ever
> tracked again.

```bash
# 1. Set up Kaggle credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle

# 2. Download all datasets
mkdir -p datasets

# Primary training dataset (same 4 classes)
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri \
  --unzip -p datasets/brain-tumor-classification-sartaj

# Additional 4-class dataset
kaggle datasets download -d bilalakgz/brain-tumor-mri-dataset \
  --unzip -p datasets/brain-tumor-bilal

# Binary tumor detection
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection \
  --unzip -p datasets/brain-mri-detection-navoneel

# 3-type classification with CSV metadata
kaggle datasets download -d jakeshbohaju/brain-tumor \
  --unzip -p datasets/figshare-brain-tumor

# Segmentation with masks (for U-Net testing)
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation \
  --unzip -p datasets/lgg-segmentation
```

### Option B: Python script

```bash
pip install opendatasets
python src/download_data.py
```

### Option C: Original notebook dataset

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset \
  --unzip -p brain-tumor-mri-dataset
```

---

## 4. Run the Training Pipeline

The notebook has been split into modular `.py` files in `src/`. You can run the full pipeline or individual steps.

### Full pipeline (train from scratch)

```bash
cd src
python run_pipeline.py
```

This runs all steps sequentially:
1. Preprocessing (CLAHE + crop + resize)
2. Data generator creation with augmentation
3. Model build + Phase 1 training (frozen base, 20 epochs) + Phase 2 fine-tuning (30 epochs)
4. Evaluation (confusion matrix, classification report, ROC curves)
5. Grad-CAM++ explainability on test samples
6. Monte Carlo Dropout uncertainty quantification
7. Test-Time Augmentation evaluation
8. TFLite INT8 export

### Skip training (load saved model)

```bash
python run_pipeline.py --skip-train
```

### Run individual steps

```bash
python -c "from config import *; from preprocessing import preprocess_all; preprocess_all()"
python -c "from config import *; from eda import *; class_distribution()"
python -c "from config import *; from model import build_model; m, e = build_model(); m.summary()"
```

### Output files produced

| File | Description |
|------|-------------|
| `model_phase1.keras` | Model after Phase 1 (frozen base) |
| `model_best.keras` | Final model after fine-tuning |
| `brain_tumor_model.tflite` | Quantized TFLite model |
| `training_history.png` | Accuracy/loss curves |
| `confusion_matrix.png` | Confusion matrix plot |
| `roc_curves.png` | Per-class ROC curves |
| `sample_predictions.png` | Sample prediction visualizations |

---

## 5. Run the FastAPI Backend

The API serves the model for the React frontend.

```bash
# From the project root (not from inside api/ — the package uses absolute imports)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Configuration

Copy `.env.example` to `.env` and adjust as needed. All settings have safe defaults for
local development.

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEYS` | *(unset)* | Comma-separated keys. When unset, authentication is disabled (local dev only). |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated CORS origins. Never use `*` in production. |
| `RATE_LIMIT_ENABLED` | `true` | Set to `false` to disable rate limiting. |
| `DATABASE_URL` | `sqlite:///./brainscan.db` | Study storage. Use a `postgresql://` URL in production. |
| `CALIBRATION_TEMP` | `1.5` | Temperature scaling applied to logits. |
| `MODEL_VERSION` | `efficientnetb1-v1` | Stamped onto every result for traceability. |
| `OOD_STATS_PATH` | `ood_stats.npz` | Fitted out-of-distribution statistics (optional). |
| `LOG_LEVEL` | `INFO` | Logging level for the structured JSON logger. |

When `API_KEYS` is set, send the key as an `X-API-Key` header, and give the frontend the
matching `VITE_API_KEY`.

### Fit the out-of-distribution detector (optional but recommended)

Without fitted statistics the API falls back to free-energy scoring. Fitting the
Mahalanobis detector on your training data makes rejection of non-brain-MRI inputs far
more reliable:

```bash
python -m scripts.fit_ood --data-dir datasets/brain-tumor-mri-dataset/Training
# writes ood_stats.npz, picked up automatically on the next API start
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, model status and build metadata |
| GET | `/metrics` | Prometheus metrics |
| POST | `/predict` | Single image classification |
| POST | `/predict/batch` | Batch classification |
| POST | `/explain/gradcam` | Grad-CAM++ heatmap |
| POST | `/explain/shap` | SHAP pixel attribution |
| POST | `/segment` | Tumor mask + volumetry |
| POST | `/analyze` | Full analysis (prediction + Grad-CAM + recommendation) — used by frontend |
| POST | `/analyze/comprehensive` | All clinical modules in one call |
| GET | `/studies` | List stored studies |
| GET | `/studies/{id}` | Retrieve a stored study |
| POST | `/studies/{id}/feedback` | Record a radiologist's review |
| GET | `/studies/{id}/report` | Download the study PDF |
| GET | `/patients/{id}/timeline` | Longitudinal growth tracking |

Full interactive documentation is at `http://localhost:8000/docs`.

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict on an image
curl -X POST http://localhost:8000/predict \
  -F "file=@datasets/brain-tumor-classification-sartaj/Testing/glioma_tumor/image_1.jpg"

# Full analysis (used by the website), grouped under a patient for timeline tracking
curl -X POST "http://localhost:8000/analyze?patient_id=PT-001" \
  -F "file=@datasets/brain-tumor-classification-sartaj/Testing/glioma_tumor/image_1.jpg"

# Download the PDF report for a study
curl -o report.pdf http://localhost:8000/studies/<study_id>/report
```

### Run the automated tests

```bash
pip install -r requirements-dev.txt
ruff check api tests scripts
pytest
```

The suite stubs the model, so TensorFlow and the trained weights are not required. This is
the same set of checks GitHub Actions runs on every push.

---

## 6. Run the React Website

The doctor-friendly website connects to the FastAPI backend.

```bash
# Terminal 1 — Start API (if not already running)
cd api && uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Start website
cd website
npm run dev
```

Open **http://localhost:5173** in your browser.

### Features
- Upload MRI scans (JPG, PNG, DICOM)
- AI classification with confidence scores
- Grad-CAM++ visual heatmap overlay
- Uncertainty estimation with review flags
- Dark mode toggle (button in navbar)
- Clinical recommendations in plain language

### Build for production

```bash
cd website
npm run build
# Output in website/dist/
```

---

## 7. Run the Multi-Dataset Test Suite

Tests your model against all 5 downloaded datasets with checkpoint/resume support.

### First run (trains model if needed, tests all datasets)

```bash
python test_all_datasets.py --fresh
```

### Resume from where it left off

If a step fails, re-run and it automatically skips passed steps:

```bash
python test_all_datasets.py
```

### Retry only failed steps

```bash
python test_all_datasets.py --retry-failed
```

### Run a single test

```bash
python test_all_datasets.py --only sartaj_test
python test_all_datasets.py --only bilal
python test_all_datasets.py --only navoneel_binary
python test_all_datasets.py --only figshare
python test_all_datasets.py --only lgg_segmentation
```

### How checkpoint/resume works

- After **each step**, results are saved to `test_checkpoint.json`
- If step 1 passes and step 2 crashes → step 1 result is preserved
- On next run → step 1 is skipped, step 2 is retried
- `--fresh` clears the checkpoint and starts over
- `--retry-failed` only re-runs steps that previously errored

### Output files

| File | Description |
|------|-------------|
| `test_checkpoint.json` | Step-by-step checkpoint (auto-saved) |
| `test_results.json` | Final results summary |

---

## 8. Run with Docker

### Full stack (API + Frontend)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Frontend | http://localhost:3000 |

### With federated learning server

```bash
docker-compose --profile federated up --build
```

### API only

```bash
docker build -t brain-tumor-api .
docker run -p 8000:8000 -v ./model_best.keras:/app/model_best.keras:ro brain-tumor-api
```

---

## 9. Run Individual Upgrade Modules

All 19 upgrades are in `upgrades/`. Each can be used independently.

| Module | Run Command | Description |
|--------|-------------|-------------|
| `vit_model.py` | `python -m upgrades.vit_model` | Vision Transformer (ViT) |
| `multitask.py` | `python -m upgrades.multitask` | Multi-task learning (class + grade + location) |
| `cnn3d.py` | `python -m upgrades.cnn3d` | 3D CNN for volumetric MRI |
| `kfold.py` | `python -m upgrades.kfold` | Stratified K-Fold cross-validation |
| `distillation.py` | `python -m upgrades.distillation` | Knowledge distillation (student model) |
| `attention_rollout.py` | `python -m upgrades.attention_rollout` | ViT attention visualization |
| `counterfactual.py` | `python -m upgrades.counterfactual` | Counterfactual explanations |
| `export_onnx.py` | `python -m upgrades.export_onnx` | ONNX export for fast inference |
| `federated.py` | `python -m upgrades.federated` | Federated learning (Flower) |
| `gan_augment.py` | `python -m upgrades.gan_augment` | GAN-based MRI augmentation |
| `multimodal.py` | `python -m upgrades.multimodal` | Image + clinical data fusion |
| `segmentation.py` | `python -m upgrades.segmentation` | U-Net tumor segmentation |
| `longitudinal.py` | `python -m upgrades.longitudinal` | Patient timeline tracking |

---

## 10. Project Structure

```
.
├── Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb  # Original notebook
├── model_best.keras                                      # Trained model
├── Dockerfile                                            # API container
├── docker-compose.yml                                    # Full stack
├── requirements-api.txt                                  # API dependencies
├── test_all_datasets.py                                  # Multi-dataset test suite
├── test_checkpoint.json                                  # Test checkpoint (auto-generated)
├── test_results.json                                     # Test results (auto-generated)
│
├── src/                        # Notebook split into modules
│   ├── config.py               #   Seeds, paths, constants
│   ├── download_data.py        #   Kaggle dataset download
│   ├── eda.py                  #   Exploratory data analysis
│   ├── preprocessing.py        #   CLAHE, crop, resize
│   ├── data_generators.py      #   Augmentation & generators
│   ├── model.py                #   EfficientNetB1 architecture
│   ├── train.py                #   Phase 1 + Phase 2 training
│   ├── evaluate.py             #   Metrics, confusion matrix, ROC
│   ├── gradcam.py              #   Grad-CAM++ implementation
│   ├── uncertainty.py          #   MC Dropout uncertainty
│   ├── tta.py                  #   Test-Time Augmentation
│   ├── gradio_app.py           #   Gradio web interface
│   ├── export.py               #   TFLite export
│   └── run_pipeline.py         #   End-to-end orchestrator
│
├── api/                        # FastAPI backend
│   ├── main.py                 #   Endpoints (/predict, /analyze, etc.)
│   ├── model_service.py        #   Model loading & inference
│   ├── schemas.py              #   Pydantic response models
│   ├── dicom_handler.py        #   DICOM file parsing
│   ├── calibration.py          #   Temperature scaling
│   └── shap_explainer.py       #   SHAP integration
│
├── upgrades/                   # 19 upgrade modules
│   ├── vit_model.py            #   Vision Transformer
│   ├── multitask.py            #   Multi-task heads
│   ├── cnn3d.py                #   3D CNN
│   ├── kfold.py                #   K-Fold CV
│   ├── distillation.py         #   Knowledge distillation
│   ├── attention_rollout.py    #   Attention visualization
│   ├── counterfactual.py       #   Counterfactual explanations
│   ├── export_onnx.py          #   ONNX export
│   ├── federated.py            #   Federated learning
│   ├── gan_augment.py          #   GAN augmentation
│   ├── multimodal.py           #   Multi-modal fusion
│   ├── segmentation.py         #   U-Net segmentation
│   └── longitudinal.py         #   Patient timeline
│
├── website/                    # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css           #   Theme (light/dark variables)
│   │   └── components/
│   │       ├── Navbar.jsx      #   Nav + dark mode toggle
│   │       ├── Hero.jsx        #   Landing section
│   │       ├── UploadAnalyze.jsx  # MRI upload + results
│   │       ├── HowItWorks.jsx  #   4-step process guide
│   │       ├── Interpret.jsx   #   Reading results guide
│   │       ├── Trust.jsx       #   Safety & accuracy info
│   │       └── Footer.jsx      #   Disclaimer footer
│   └── package.json
│
└── datasets/                   # Downloaded test datasets
    ├── brain-tumor-classification-sartaj/   # 4 classes (drop-in)
    ├── brain-tumor-bilal/                   # 4 classes
    ├── brain-mri-detection-navoneel/        # Binary (yes/no)
    ├── figshare-brain-tumor/               # 3 types + CSV
    └── lgg-segmentation/                   # FLAIR + masks
```

---

## 11. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | `pip install tensorflow` |
| `CUDA out of memory` | Reduce batch size in `data_generators.py` (default 32) |
| `Kaggle 403 Forbidden` | Check `~/.kaggle/kaggle.json` credentials and permissions (`chmod 600`) |
| API returns `Model not found` | Ensure `model_best.keras` exists in project root. Train first or use `--skip-train` |
| Website can't connect to API | Start API first (`uvicorn`), check CORS (port 5173 is whitelisted) |
| Test suite step fails | Re-run `python test_all_datasets.py` — it resumes from last checkpoint |
| Dark mode not toggling | Clear browser localStorage: `localStorage.removeItem('theme')` |
| Docker build fails | Ensure `model_best.keras` exists before building (it's mounted as a volume) |
| `imutils` import error | `pip install imutils` |
| Mixed precision warnings | Safe to ignore — model uses `mixed_float16` for GPU speedup |
