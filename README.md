# Brain Tumor Classification Using Deep Learning & Grad-CAM

A Deep Learning project for brain tumor classification using MRI images. This project uses Convolutional Neural Networks (CNNs) to classify brain tumors and integrates Grad-CAM (Gradient-weighted Class Activation Mapping) for explainable AI and visual interpretation of predictions.

## Features

- Brain tumor classification using MRI scans
- Deep Learning-based image classification
- Image preprocessing and normalization
- Model training and evaluation
- Prediction on MRI images
- Explainable AI using Grad-CAM
- Heatmap visualization of tumor regions
- Performance evaluation and visualization

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Load MRI brain tumor image dataset
2. Preprocess and normalize images
3. Split data into training and testing sets
4. Build and train CNN model
5. Evaluate model performance
6. Predict tumor classes
7. Generate Grad-CAM heatmaps
8. Visualize important regions influencing predictions

## Model Features

The model is designed to:

- Learn patterns from MRI brain scans
- Classify brain tumors using Deep Learning
- Highlight important image regions using Grad-CAM
- Improve model interpretability with explainable AI

## Folder Structure

```bash
├── Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb
├── dataset/
├── saved_model/
├── outputs/
├── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
Brain_Tumor_Classification_Using_DL_&_GradCAM.ipynb
```

## Example Workflow

### Input
MRI brain scan image

### Output
- Predicted brain tumor class
- Confidence score
- Grad-CAM heatmap visualization showing important regions

## Grad-CAM Visualization

Grad-CAM helps explain model predictions by highlighting regions of the MRI image that strongly influenced the classification decision.

This improves:

- Model transparency
- Explainability
- Medical AI interpretability

## Evaluation Metrics

Model performance may be evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Future Improvements

- Better CNN architectures (EfficientNet, ResNet)
- Hyperparameter tuning
- Web app deployment using Flask or Streamlit
- Real-time MRI image prediction
- Improved explainability methods

## Author

Adithya C
