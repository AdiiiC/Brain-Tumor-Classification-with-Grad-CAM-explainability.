# Brain Tumor Classification Using Deep Learning & Grad-CAM

A Deep Learning-based Brain Tumor Classification project that uses MRI images to classify brain tumors and applies Grad-CAM (Gradient-weighted Class Activation Mapping) for explainable AI visualization. The project uses Convolutional Neural Networks (CNNs) to learn image patterns and visually explain predictions by highlighting important regions in MRI scans.

---

## Features

- Brain tumor classification using MRI images
- Deep Learning-based image classification using CNNs
- Automated dataset download using `opendatasets`
- MRI image preprocessing and normalization
- Model training and evaluation
- Grad-CAM visualization for explainable AI
- Prediction on MRI scans
- Visualization of important tumor regions
- Model performance analysis

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- opendatasets
- Kaggle API

---

## Dataset

This project uses the **Brain Tumor MRI Dataset** from Kaggle.

Dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset is automatically downloaded using:

```python
import opendatasets as od

od.download(
    "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data"
)
```

### Dataset Classes

The dataset contains MRI scans categorized into:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Project Workflow

1. Download MRI dataset from Kaggle
2. Load and preprocess MRI images
3. Normalize and prepare training data
4. Train CNN model
5. Evaluate model performance
6. Predict tumor classes
7. Generate Grad-CAM heatmaps
8. Visualize important regions influencing predictions

---

## Project Structure

```bash
Brain-Tumor-Classification/
│── Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb
│── README.md
│── requirements.txt
│── .gitignore
│── outputs/                  # Grad-CAM outputs & screenshots (optional)
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Kaggle API Setup

To download the dataset, you need Kaggle API credentials.

### Step 1: Create Kaggle API Token

1. Login to Kaggle
2. Go to **Account Settings**
3. Click **Create New API Token**
4. Download `kaggle.json`

### Step 2: Configure Kaggle API

Move `kaggle.json` to:

**Windows**

```text
C:\Users\YourUsername\.kaggle\
```

**Linux / Mac**

```bash
~/.kaggle/
```

---

## Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb
```

Run all cells.

---

## Example Output

### Input
MRI brain scan image

### Output
- Predicted tumor type
- Confidence score
- Grad-CAM heatmap showing important regions influencing prediction

---

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) helps explain model predictions by highlighting regions of MRI scans that influenced the model’s decision.

Benefits:

- Model explainability
- Better interpretability
- Medical AI transparency
- Visualization of tumor-relevant regions

---

## Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## requirements.txt

Example dependencies:

```txt
tensorflow
keras
numpy
pandas
matplotlib
opencv-python
scikit-learn
seaborn
jupyter
opendatasets
kaggle
```

---

## Future Improvements

- Better CNN architectures (EfficientNet, ResNet)
- Hyperparameter tuning
- Real-time MRI prediction system
- Web deployment using Flask or Streamlit
- Improved explainability techniques

---

## Author

Your Name
