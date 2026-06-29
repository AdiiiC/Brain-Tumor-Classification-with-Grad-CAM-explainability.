"""
Main pipeline: runs the full training + evaluation + export workflow.
Execute this to reproduce the entire notebook pipeline end-to-end.

Usage:
    python run_pipeline.py              # Full pipeline
    python run_pipeline.py --skip-train # Skip training, load saved model
"""
import argparse

import tensorflow as tf

from config import CLASS_NAMES
from preprocessing import preprocess_all
from data_generators import create_generators, visualize_augmented
from model import build_model
from train import compute_class_weights, train_phase1, train_phase2, plot_training_history
from evaluate import (
    evaluate_model, generate_predictions, plot_confusion_matrix,
    print_classification_report, plot_roc_curves,
    predict_individual_images, visualize_predictions
)
from gradcam import run_gradcam_all_classes
from uncertainty import demo_uncertainty
from tta import evaluate_tta
from export import export_tflite, verify_tflite


def main(skip_train=False):
    # Step 1: Preprocessing
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing images")
    print("=" * 60)
    preprocess_all()

    # Step 2: Data generators
    print("\n" + "=" * 60)
    print("STEP 2: Creating data generators")
    print("=" * 60)
    train_data, valid_data, test_data = create_generators()
    visualize_augmented(train_data)

    # Step 3: Model
    if skip_train:
        print("\n" + "=" * 60)
        print("STEP 3: Loading pre-trained model")
        print("=" * 60)
        model = tf.keras.models.load_model('model_best.keras')
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Building & training model")
        print("=" * 60)
        model, effenet = build_model(freeze_base=True)
        class_weight_dict = compute_class_weights(train_data)
        history_p1 = train_phase1(model, train_data, valid_data, class_weight_dict)
        history_p2 = train_phase2(model, effenet, train_data, valid_data, class_weight_dict)
        plot_training_history(history_p1, history_p2)

    # Step 4: Evaluation
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)
    evaluate_model(model, train_data, test_data)
    y_test, y_test_hat = generate_predictions(model, test_data)
    plot_confusion_matrix(y_test, y_test_hat)
    print_classification_report(y_test, y_test_hat)
    plot_roc_curves(model, test_data, y_test)
    images, prediction, original, score = predict_individual_images(model)
    visualize_predictions(images, prediction, original)

    # Step 5: Explainability
    print("\n" + "=" * 60)
    print("STEP 5: Grad-CAM++ Explainability")
    print("=" * 60)
    run_gradcam_all_classes(model)

    # Step 6: Uncertainty
    print("\n" + "=" * 60)
    print("STEP 6: Monte Carlo Dropout Uncertainty")
    print("=" * 60)
    demo_uncertainty(model)

    # Step 7: TTA
    print("\n" + "=" * 60)
    print("STEP 7: Test-Time Augmentation")
    print("=" * 60)
    evaluate_tta(model)

    # Step 8: Export
    print("\n" + "=" * 60)
    print("STEP 8: TFLite Export")
    print("=" * 60)
    tflite_path = export_tflite(model)
    verify_tflite(tflite_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Classification Pipeline")
    parser.add_argument('--skip-train', action='store_true', help='Skip training, load saved model')
    args = parser.parse_args()
    main(skip_train=args.skip_train)
