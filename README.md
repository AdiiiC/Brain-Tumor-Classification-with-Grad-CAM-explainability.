# Brain Tumor Classification with Grad-CAM++ Explainability

An end-to-end brain tumor classification system using EfficientNetB1 with Grad-CAM++ visual explainability, Monte Carlo Dropout uncertainty estimation, and a doctor-friendly React clinical interface backed by a FastAPI REST API.

Classifies brain MRI scans into 4 categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

---

## Key Features

- **EfficientNetB1 with Transfer Learning** — 2-phase training (frozen base → fine-tuning top 30 layers)
- **Grad-CAM++ Explainability** — visual heatmaps showing which brain regions influenced the prediction
- **Monte Carlo Dropout** — uncertainty quantification (30 stochastic forward passes) with auto-flagging for specialist review
- **Test-Time Augmentation (TTA)** — boosts accuracy by averaging predictions over augmented views
- **Calibrated Confidence** — temperature scaling so reported confidence matches actual accuracy
- **DICOM Support** — accepts clinical DICOM files directly from imaging equipment
- **Doctor-Friendly React Website** — clean clinical UI with dark mode, upload-and-analyze workflow, visual explanations
- **FastAPI Backend** — REST endpoints for prediction, batch inference, Grad-CAM, SHAP, and full analysis
- **Multi-Dataset Test Suite** — checkpoint/resume system that tests across 5 datasets
- **19 Upgrade Modules** — ViT, 3D CNN, U-Net segmentation, federated learning, knowledge distillation, GAN augmentation, and more
- **Docker Deployment** — single-command deployment with docker-compose

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (:5173)                     │
│  Upload MRI → View Results + Grad-CAM → Dark/Light Mode     │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /analyze
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI Backend (:8000)                     │
│  /predict  /predict/batch  /explain/gradcam  /explain/shap   │
│  /analyze (full pipeline)  /health                           │
├──────────────────────────────────────────────────────────────┤
│  ModelService: EfficientNetB1 + Grad-CAM++ + MC Dropout      │
│  DICOM Handler │ Temperature Calibration │ SHAP Explainer    │
└──────────────────────────────────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  model_best.keras (26 MB) │
         │  EfficientNetB1, 4 classes│
         │  240×240 input, softmax   │
         └───────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Model | TensorFlow/Keras, EfficientNetB1, mixed_float16 precision |
| Explainability | Grad-CAM++, SHAP DeepExplainer, Attention Rollout |
| Backend | FastAPI, Uvicorn, Pydantic, pydicom |
| Frontend | React 18, Vite, CSS custom properties (light/dark theme) |
| Testing | scikit-learn metrics, multi-dataset checkpoint/resume |
| Deployment | Docker, docker-compose |
| Data | OpenCV, CLAHE preprocessing, ImageDataGenerator augmentation |

---

## Project Structure

```
.
├── Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb  # Original notebook
├── test_all_datasets.py          # Multi-dataset test suite with checkpoint/resume
├── Dockerfile                    # API container
├── docker-compose.yml            # Full stack (API + frontend + federated)
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
├── api/                          # FastAPI backend
│   ├── main.py                   #   REST endpoints
│   ├── model_service.py          #   Inference, Grad-CAM++, MC Dropout, TTA
│   ├── schemas.py                #   Pydantic response models
│   ├── dicom_handler.py          #   DICOM file parsing
│   ├── calibration.py            #   Temperature scaling
│   └── shap_explainer.py         #   SHAP integration
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
├── website/                      # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx / App.css
│   │   ├── index.css             #   Light/dark theme variables
│   │   └── components/
│   │       ├── Navbar.jsx        #   Navigation + dark mode toggle
│   │       ├── Hero.jsx          #   Landing section
│   │       ├── UploadAnalyze.jsx #   MRI upload → results display
│   │       ├── HowItWorks.jsx    #   4-step process guide
│   │       ├── Interpret.jsx     #   Guide to reading results
│   │       ├── Trust.jsx         #   Safety, accuracy, limitations
│   │       └── Footer.jsx        #   Medical disclaimer
│   └── package.json
│
└── datasets/                     # Downloaded test datasets (gitignored)
    ├── brain-tumor-classification-sartaj/
    ├── brain-tumor-bilal/
    ├── brain-mri-detection-navoneel/
    ├── figshare-brain-tumor/
    └── lgg-segmentation/
```

---

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url>
cd Brain-Tumor-Classification-with-Grad-CAM-explainability.
python -m venv .venv && source .venv/bin/activate
pip install tensorflow opencv-python scikit-learn imutils seaborn tqdm Pillow
pip install -r requirements-api.txt

# 2. Train the model
cd src && python run_pipeline.py

# 3. Start the API
cd ../api && uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Start the website (new terminal)
cd website && npm install && npm run dev
# Open http://localhost:5173
```

See [SETUP.md](SETUP.md) for the complete guide including dataset downloads, Docker deployment, and test suite usage.

---

## Dataset

Primary: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

| Class | Training | Test |
|-------|----------|------|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |

Additional test datasets (see [SETUP.md](SETUP.md) for download commands):
- **Sartaj** — 4-class, drop-in compatible
- **Bilal** — 4-class cross-dataset evaluation
- **Navoneel** — binary tumor detection
- **Figshare** — binary with CSV metadata
- **LGG Segmentation** — FLAIR MRI with segmentation masks

---

## Training

Two-phase transfer learning on EfficientNetB1 (ImageNet weights):

| Phase | Layers | Learning Rate | Epochs | Purpose |
|-------|--------|--------------|--------|---------|
| 1 | Head only (base frozen) | 1e-4 | 20 | Learn classifier |
| 2 | Top 30 + head | 1e-5 | 30 | Fine-tune features |

Includes class weighting, CLAHE preprocessing, augmentation (rotation, zoom, shift, flip, brightness), and early stopping with LR reduction.

---

## API Endpoints

```bash
# Start the API
cd api && uvicorn main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single image → class + confidence |
| `POST` | `/predict/batch` | Multiple images |
| `POST` | `/explain/gradcam` | Grad-CAM++ heatmap |
| `POST` | `/explain/shap` | SHAP pixel attribution |
| `POST` | `/analyze` | Full analysis (prediction + Grad-CAM + recommendation) |

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

## Docker

```bash
# Full stack (API on :8000 + frontend on :3000)
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
