"""
Download dataset from Kaggle.
Corresponds to Notebook Cells 1-2.
"""
import subprocess
import sys


def install_deps():
    """Install required packages."""
    packages = [
        "opendatasets", "tensorflow", "opencv-python", "scikit-learn",
        "imutils", "gradio", "tf2onnx", "seaborn", "tqdm", "Pillow", "scikit-image"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)


def download_dataset():
    """Download brain tumor MRI dataset from Kaggle."""
    import opendatasets as od
    od.download("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data")


if __name__ == "__main__":
    install_deps()
    download_dataset()
