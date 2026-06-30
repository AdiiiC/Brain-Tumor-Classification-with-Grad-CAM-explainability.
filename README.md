# Brain Tumor Classification with Grad-CAM++ Explainability

An end-to-end brain tumor classification system using EfficientNetB1 with Grad-CAM++ visual explainability, Monte Carlo Dropout uncertainty estimation, and a clinical React interface backed by a FastAPI REST API with 5 specialized clinical modules.

Classifies brain MRI scans into 4 categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

**Live Demo:** [Frontend (Vercel)](https://brain-tumor-classification-with-grad-cam-explainability.vercel.app) | [API (Render)](https://brain-tumor-classification-2l6r.onrender.com/health)

---

## Key Features

- **EfficientNetB1 with Transfer Learning** — 2-phase training (frozen base → fine-tuning top 30 layers)
- **Grad-CAM++ Explainability** — visual heatmaps showing which brain regions influenced the prediction
- **Monte Carlo Dropout** — uncertainty quantification (50 stochastic forward passes) with auto-flagging for specialist review
- **Test-Time Augmentation (TTA)** — weighted TTA boosts accuracy by averaging predictions over augmented views
- **Calibrated Confidence** — temperature scaling (T=1.5) so reported confidence matches actual accuracy
- **DICOM Support** — accepts clinical DICOM files directly from imaging equipment
- **Multi-Sequence MRI Training** — trained on T1, T1CE, T2, and FLAIR sequences from BraTS 2021
- **Class-Balanced Training** — offline augmentation oversampling to equalize all 4 classes
- **5 Clinical Modules:**
  - **Image Quality Assessment** — resolution, blur, SNR, compression, brain coverage scoring
  - **MRI Sequence Detection** — auto-detects T1/T1CE/T2/FLAIR/DWI from image statistics or DICOM metadata
  - **WHO Tumor Grading** — Grade I-IV estimation using a dedicated MobileNetV2 grade classifier + image features
  - **Small Tumor Detection** — sliding-window patch-based detection using a trained MobileNetV2 patch classifier
  - **Pediatric Support** — Bayesian re-weighting of predictions based on age-specific tumor priors
- **Grade Classifier** — MobileNetV2 (240×240) trained on HGG/LGG data for binary glioma grading
- **Patch Classifier** — MobileNetV2 (120×120) trained on tumor/clean patches for small tumor detection
- **Doctor-Friendly React Website** — clinical UI with dark mode, upload-and-analyze, visual explanations, live results page
- **FastAPI Backend** — 16 REST endpoints for prediction, explainability, quality assessment, and comprehensive analysis
- **Multi-Dataset Test Suite** — checkpoint/resume system that tests across 5 datasets (21,732 images)
- **19 Upgrade Modules** — ViT, 3D CNN, U-Net segmentation, federated learning, knowledge distillation, GAN augmentation, and more
- **Docker + Cloud Deployment** — Docker, Vercel (frontend), Render (backend)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vercel)                        │
│  Upload MRI → Results + Grad-CAM → Live Demo → Dark/Light Mode  │
└──────────────────────┬───────────────────────────────────────────┘
                       │ POST /analyze, /assess/*, /detect/*
┌──────────────────────▼───────────────────────────────────────────┐
│                   FastAPI Backend (Render)                        │
│  /predict  /predict/batch  /explain/gradcam  /explain/shap       │
│  /analyze  /analyze/comprehensive  /health                       │
│  /assess/quality  /assess/sequence  /assess/grade                │
│  /assess/pediatric  /detect/small-tumors                         │
├──────────────────────────────────────────────────────────────────┤
│  ModelService     │ ImageQuality  │ SequenceDetector             │
│  (EfficientNetB1) │ (blur/SNR/    │ (T1/T1CE/T2/                │
│                   │  resolution)  │  FLAIR/DWI)                  │
│  TumorGrading     │ SmallTumor    │ PediatricSupport             │
│  (MobileNetV2     │ (MobileNetV2  │ (Bayesian                   │
│   grade model)    │  patch model) │  re-weighting)               │
└──────────────────────────────────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────────────┐
         │  model_best.keras (26 MB)          │
         │  grade_classifier.keras (MobileV2) │
         │  patch_classifier.keras (MobileV2) │
         └────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Model | TensorFlow/Keras, EfficientNetB1 (main), MobileNetV2 (grade + patch), mixed_float16 |
| Explainability | Grad-CAM++, SHAP DeepExplainer, Attention Rollout |
| Backend | FastAPI, Uvicorn, Pydantic, pydicom |
| Frontend | React 18, Vite, CSS custom properties (light/dark theme) |
| Testing | scikit-learn metrics, multi-dataset checkpoint/resume (5 datasets, 21,732 images) |
| Deployment | Docker, Vercel (frontend), Render (backend) |
| Data | OpenCV, CLAHE preprocessing, offline augmentation oversampling |

---

## Test Results

Tested across 5 independent datasets (21,732 total images):

| Dataset | Images | Task | Accuracy |
|---------|--------|------|----------|
| Standard MRI Scans | 1,311 | 4-class tumor type | 84.26% |
| Hospital MRI Collection | 2,870 | 4-class tumor type | 85.50% |
| Clinical Detection Set | 253 | Tumor vs. healthy | 83.40% |
| Confirmed Tumor Scans | 3,064 | Tumor type identification | 100.00% |
| FLAIR Sequence MRIs | 3,929 | Tumor vs. healthy (FLAIR) | 77.50% |

Training data: 21,732 images balanced across 4 classes using offline augmentation oversampling, including T1, T1CE, T2, and FLAIR MRI sequences from BraTS 2021.

---

## Project Structure

```
.
├── Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb  # Original notebook
├── test_all_datasets.py          # Multi-dataset test suite with checkpoint/resume
├── model_best.keras              # Main EfficientNetB1 model (4-class)
├── grade_classifier.keras        # MobileNetV2 grade classifier (HGG/LGG)
├── patch_classifier.keras        # MobileNetV2 patch classifier (tumor/clean)
├── Dockerfile                    # API container
├── docker-compose.yml            # Full stack deployment
├── render.yaml                   # Render backend config
├── requirements-api.txt          # API Python dependencies
├── SETUP.md                      # Detailed step-by-step run guide
│
├── src/                          # Notebook split into modules
│   ├── config.py                 #   Reproducibility, paths, constants
│   ├── download_data.py          #   Kaggle dataset download
│   ├── eda.py                    #   Class distribution, sample visualization
│   ├── preprocessing.py          #   CLAHE enhancement, brain cropping, resizing
│   ├── data_generators.py        #   Augmentation pipelines (train/val/test)
│   ├── model.py                  #   EfficientNetB1 architecture
│   ├── train.py                  #   Phase 1 (frozen) + Phase 2 (fine-tune)
│   ├── evaluate.py               #   Confusion matrix, ROC curves, reports
│   ├── gradcam.py                #   Grad-CAM++ implementation
│   ├── uncertainty.py            #   Monte Carlo Dropout
│   ├── tta.py                    #   Test-Time Augmentation
│   ├── gradio_app.py             #   Gradio web interface
│   ├── export.py                 #   TFLite INT8 export
│   └── run_pipeline.py           #   End-to-end orchestrator
│
├── api/                          # FastAPI backend (16 endpoints)
│   ├── main.py                   #   REST endpoints + clinical modules
│   ├── model_service.py          #   Inference, Grad-CAM++, MC Dropout, TTA
│   ├── schemas.py                #   Pydantic response models
│   ├── dicom_handler.py          #   DICOM file parsing
│   ├── calibration.py            #   Temperature scaling (T=1.5)
│   ├── shap_explainer.py         #   SHAP integration
│   ├── image_quality.py          #   Image quality assessment module
│   ├── sequence_detector.py      #   MRI sequence auto-detection
│   ├── tumor_grading.py          #   WHO Grade I-IV estimation
│   ├── small_tumor_detector.py   #   Patch-based small tumor detection
│   └── pediatric_support.py      #   Bayesian pediatric re-weighting
│
├── upgrades/                     # Advanced upgrade modules
│   ├── vit_model.py              #   Vision Transformer
│   ├── multitask.py              #   Multi-task (class + grade + location)
│   ├── cnn3d.py                  #   3D CNN for volumetric MRI
│   ├── kfold.py                  #   Stratified K-Fold cross-validation
│   ├── distillation.py           #   Knowledge distillation
│   ├── attention_rollout.py      #   ViT attention visualization
│   ├── counterfactual.py         #   Counterfactual explanations
│   ├── export_onnx.py            #   ONNX export
│   ├── federated.py              #   Federated learning (Flower + DP)
│   ├── gan_augment.py            #   DCGAN synthetic MRI generation
│   ├── multimodal.py             #   Image + clinical data fusion
│   ├── segmentation.py           #   U-Net tumor segmentation
│   └── longitudinal.py           #   Patient timeline tracking
│
├── website/                      # React frontend (Vite → Vercel)
│   ├── vercel.json               #   Vercel deployment config
│   ├── .env.production           #   Production API URL
│   ├── public/samples.json       #   Base64 sample images for live demo
│   ├── src/
│   │   ├── App.jsx / App.css
│   │   ├── index.css             #   Light/dark theme variables
│   │   └── components/
│   │       ├── Navbar.jsx        #   Navigation + dark mode toggle
│   │       ├── Hero.jsx          #   Landing section
│   │       ├── UploadAnalyze.jsx #   MRI upload → results display
│   │       ├── HowItWorks.jsx    #   4-step process guide
│   │       ├── Interpret.jsx     #   Guide to reading results
│   │       ├── Results.jsx       #   Test results + live API demo
│   │       ├── Trust.jsx         #   Safety, accuracy, resolved limitations
│   │       └── Footer.jsx        #   Medical disclaimer
│   └── package.json
│
└── datasets/                     # Downloaded test datasets (gitignored)
    ├── brain-tumor-classification-sartaj/
    ├── brain-tumor-bilal/
    ├── brain-mri-detection-navoneel/
    ├── figshare-brain-tumor/
    ├── lgg-segmentation/
    └── brats2021-2d/ (T1, T1CE, T2, FLAIR sequences)
```

---

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/AdiiiC/Brain-Tumor-Classification-with-Grad-CAM-explainability..git
cd Brain-Tumor-Classification-with-Grad-CAM-explainability.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt

# 2. Start the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Start the website (new terminal)
cd website && npm install && npm run dev
# Open http://localhost:5173
```

See [SETUP.md](SETUP.md) for the complete guide including dataset downloads, Docker deployment, training, and test suite usage.

---

## Datasets

Trained and tested on 21,732 images from 6 sources:

| Source | Images | Type | Purpose |
|--------|--------|------|---------|
| Standard MRI collection | 7,023 | 4-class labeled MRI | Primary training + test |
| Hospital MRI set | 2,870 | 4-class labeled MRI | Cross-dataset test |
| Clinical detection set | 253 | Binary (tumor/healthy) | Binary detection test |
| Confirmed tumor scans | 3,064 | Labeled with CSV metadata | Metadata-based test |
| FLAIR MRI collection | 3,929 | FLAIR with segmentation masks | Sequence-specific test |
| BraTS 2021 (2D slices) | 2,000 | T1/T1CE/T2/FLAIR (500 each) | Multi-sequence training |

All classes balanced via offline augmentation oversampling before training.

---

## Training

Two-phase transfer learning on EfficientNetB1 (ImageNet weights):

| Phase | Layers | Learning Rate | Epochs | Purpose |
|-------|--------|--------------|--------|---------|
| 1 | Head only (base frozen) | 1e-4 | 20 | Learn classifier |
| 2 | Top 30 + head | 1e-5 | 30 | Fine-tune features |

Additional models trained after the main model:
- **Grade Classifier** (MobileNetV2, 240×240) — binary HGG/LGG classification for WHO grading
- **Patch Classifier** (MobileNetV2, 120×120) — binary tumor/clean classification for small tumor detection

Post-training calibration: temperature scaling (T=1.5) applied to softmax outputs.

---

## API Endpoints

```bash
# Start the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single image → class + confidence |
| `POST` | `/predict/batch` | Multiple images |
| `POST` | `/explain/gradcam` | Grad-CAM++ heatmap |
| `POST` | `/explain/shap` | SHAP pixel attribution |
| `POST` | `/analyze` | Full analysis (prediction + Grad-CAM + recommendation) |
| `POST` | `/assess/quality` | Image quality score (resolution, blur, SNR, compression) |
| `POST` | `/assess/sequence` | Auto-detect MRI sequence (T1/T1CE/T2/FLAIR/DWI) |
| `POST` | `/assess/grade` | WHO tumor grade estimation (Grade I-IV) |
| `POST` | `/assess/pediatric` | Pediatric assessment with Bayesian re-weighting |
| `POST` | `/detect/small-tumors` | Small tumor detection via patch-based sliding window |
| `POST` | `/analyze/comprehensive` | All modules combined in one call |

---

## Multi-Dataset Test Suite

```bash
python test_all_datasets.py --fresh         # Train + test all from scratch
python test_all_datasets.py                 # Resume from last checkpoint
python test_all_datasets.py --retry-failed  # Re-run only failed steps
python test_all_datasets.py --only bilal    # Run a single test
```

Checkpoint/resume: results are saved to `test_checkpoint.json` after **each** step. If step 3 crashes, steps 1-2 are preserved and step 3 retries on the next run.

---

## Deployment

| Component | Platform | Config |
|-----------|----------|--------|
| Frontend | Vercel | `website/vercel.json` — auto-detects Vite, serves static build |
| Backend | Render | `render.yaml` + `Dockerfile` — Docker-based, free tier |

The frontend reads the API URL from `VITE_API_URL` environment variable (set in `website/.env.production` for production builds, falls back to `localhost:8000` for local development).

```bash
# Docker (local)
docker-compose up --build

# With federated learning server (:8080)
docker-compose --profile federated up --build
```

---

## Upgrade Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | `vit_model.py` | Vision Transformer with patch embeddings |
| 2 | `multitask.py` | Multi-task learning (class + grade + location) |
| 3 | `cnn3d.py` | 3D CNN for volumetric MRI (NIfTI/DICOM series) |
| 4 | `kfold.py` | Stratified K-Fold CV with confidence intervals |
| 5 | `distillation.py` | Knowledge distillation to lightweight student model |
| 6 | `attention_rollout.py` | ViT attention map visualization |
| 7 | `counterfactual.py` | Occlusion-based counterfactual explanations |
| 8 | `export_onnx.py` | ONNX export for fast cross-platform inference |
| 9 | `federated.py` | Federated learning (Flower) with differential privacy |
| 10 | `gan_augment.py` | DCGAN for synthetic MRI generation |
| 11 | `multimodal.py` | Image + clinical tabular data fusion |
| 12 | `segmentation.py` | U-Net tumor segmentation with dice loss |
| 13 | `longitudinal.py` | Patient timeline tracking, growth rate, alerts |

---

## License

This project is for research and educational purposes. The AI system is a clinical decision-support tool — not a certified medical device. All results must be reviewed by a qualified healthcare professional.
