"""
Training pipeline: Phase 1 (frozen base) + Phase 2 (fine-tuning).
Corresponds to Notebook Cells 28-31.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from config import CLASSES
from model import build_model, unfreeze_top_layers
from data_generators import create_generators


def compute_class_weights(train_data):
    """Compute balanced class weights."""
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print("Class weights:", {CLASSES[k]: round(v, 3) for k, v in class_weight_dict.items()})
    return class_weight_dict


def train_phase1(model, train_data, valid_data, class_weight_dict, epochs=20):
    """Phase 1: Train classifier head only (frozen base)."""
    print("=== Phase 1: Training classifier head ===")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint('model_phase1.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5,
                      mode='max', verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                          patience=2, min_delta=0.001, mode='max', verbose=1)
    ]

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=valid_data,
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    return history


def train_phase2(model, effenet, train_data, valid_data, class_weight_dict, epochs=30):
    """Phase 2: Fine-tune top 30 EfficientNet layers."""
    print("=== Phase 2: Fine-tuning top layers ===")
    unfreeze_top_layers(effenet, n_layers=30)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint('model_best.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7,
                      mode='max', verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                          patience=2, min_delta=0.001, mode='max', verbose=1)
    ]

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=valid_data,
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    return history


def plot_training_history(history_p1, history_p2):
    """Plot combined accuracy and loss curves from both training phases."""
    train_acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
    val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
    train_loss = history_p1.history['loss'] + history_p2.history['loss']
    val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']
    epochs = range(1, len(train_acc) + 1)
    phase_split = len(history_p1.history['accuracy'])

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0].plot(epochs, train_acc, 'g-o', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'y-o', label='Validation Accuracy')
    ax[0].axvline(x=phase_split, color='red', linestyle='--', label='Fine-tune start')
    ax[0].set_title('Model Accuracy — Phase 1 & 2')
    ax[0].legend(loc='lower right')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'y-o', label='Validation Loss')
    ax[1].axvline(x=phase_split, color='red', linestyle='--', label='Fine-tune start')
    ax[1].set_title('Model Loss — Phase 1 & 2')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    train_data, valid_data, test_data = create_generators()
    model, effenet = build_model(freeze_base=True)
    class_weight_dict = compute_class_weights(train_data)

    history_p1 = train_phase1(model, train_data, valid_data, class_weight_dict)
    history_p2 = train_phase2(model, effenet, train_data, valid_data, class_weight_dict)
    plot_training_history(history_p1, history_p2)
