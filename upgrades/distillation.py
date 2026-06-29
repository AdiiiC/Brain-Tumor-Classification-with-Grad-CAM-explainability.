"""
Upgrade #5 — Knowledge Distillation.

Train a smaller student model from the EfficientNetB1 teacher.
The student learns to mimic the teacher's soft predictions (dark knowledge),
achieving comparable accuracy with 5-10× fewer parameters.

Ideal for edge/mobile deployment where model size matters.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


def build_student_model(img_size=240, num_classes=4) -> Model:
    """
    Lightweight student model — ~500K params vs teacher's ~7.8M.

    Architecture: Simple CNN with depthwise-separable convolutions.
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise-separable blocks
    for filters in [64, 128, 256]:
        x = layers.DepthwiseConv2D(3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs, outputs, name="Student_Lightweight")
    return model


class DistillationLoss(tf.keras.losses.Loss):
    """
    Knowledge Distillation loss = α * KL(soft_teacher || soft_student) + (1-α) * CE(hard_label, student)

    Temperature T controls how "soft" the teacher's predictions are.
    Higher T → more information from similar classes.
    """

    def __init__(self, temperature=4.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def call(self, y_true_and_teacher, y_pred):
        # y_true_and_teacher is concatenated: [one_hot_label, teacher_logits]
        num_classes = y_pred.shape[-1]
        y_true = y_true_and_teacher[:, :num_classes]
        teacher_logits = y_true_and_teacher[:, num_classes:]

        # Soft targets
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
        student_soft = tf.nn.softmax(y_pred / self.temperature)

        # KL divergence (distillation loss)
        kl_loss = tf.keras.losses.KLDivergence()(teacher_soft, student_soft) * (self.temperature ** 2)

        # Hard target loss (standard cross-entropy)
        ce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, tf.nn.softmax(y_pred))

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


def distill(
    teacher_model,
    train_data,
    val_data,
    temperature=4.0,
    alpha=0.7,
    epochs=30,
    lr=1e-3,
) -> tuple[Model, dict]:
    """
    Train student via knowledge distillation from teacher.

    Args:
        teacher_model: Trained EfficientNetB1 model
        train_data: Training generator
        val_data: Validation generator
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation loss vs hard loss
        epochs: Training epochs
        lr: Learning rate

    Returns:
        (student_model, training_history)
    """
    student = build_student_model()
    num_classes = 4

    # Get teacher logit model (before softmax)
    teacher_logit_model = Model(teacher_model.input, teacher_model.layers[-1].output)

    # Custom training loop
    optimizer = Adam(learning_rate=lr)
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    kl_loss_fn = tf.keras.losses.KLDivergence()

    best_val_acc = 0
    history = {"loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        epoch_loss = []

        for x_batch, y_batch in train_data:
            # Teacher predictions (no gradient)
            teacher_logits = teacher_logit_model(x_batch, training=False)
            teacher_soft = tf.nn.softmax(teacher_logits / temperature)

            with tf.GradientTape() as tape:
                # Student predictions
                student_pred = student(x_batch, training=True)

                # Distillation loss
                student_soft = tf.nn.softmax(student_pred / temperature)
                distill_loss = kl_loss_fn(teacher_soft, student_soft) * (temperature ** 2)

                # Hard label loss
                hard_loss = ce_loss_fn(y_batch, student_pred)

                # Combined
                total_loss = alpha * distill_loss + (1 - alpha) * hard_loss

            grads = tape.gradient(total_loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            epoch_loss.append(float(total_loss))

        # Validation
        val_correct = 0
        val_total = 0
        for x_val, y_val in val_data:
            pred = student(x_val, training=False)
            val_correct += tf.reduce_sum(
                tf.cast(tf.argmax(pred, 1) == tf.argmax(y_val, 1), tf.float32)
            ).numpy()
            val_total += len(x_val)

        val_acc = val_correct / val_total
        avg_loss = np.mean(epoch_loss)
        history["loss"].append(avg_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            student.save("model_student_best.keras")

    # Print comparison
    teacher_params = teacher_model.count_params()
    student_params = student.count_params()
    print(f"\nTeacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}")
    print(f"Compression:    {teacher_params/student_params:.1f}×")
    print(f"Student accuracy: {best_val_acc*100:.2f}%")

    return student, history
