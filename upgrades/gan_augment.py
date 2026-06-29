"""
Upgrade #16 — GAN-Based Data Augmentation.

Uses a Deep Convolutional GAN (DCGAN) to generate synthetic MRI images.
Particularly valuable for rare tumor types with limited real samples.
Synthetic images supplement real data during training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_generator(latent_dim=128, img_size=240) -> Model:
    """
    Generator network: random noise → synthetic MRI image.

    Architecture: Dense → Reshape → Upsampling Conv blocks → 240×240×3
    """
    # Start with a small spatial resolution and upsample
    init_size = img_size // 16  # 15

    inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(init_size * init_size * 256, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((init_size, init_size, 256))(x)

    # Upsample: 15 → 30
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 30 → 60
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 60 → 120
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 120 → 240
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Output: 240×240×3
    outputs = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)

    return Model(inputs, outputs, name="MRI_Generator")


def build_discriminator(img_size=240) -> Model:
    """
    Discriminator network: MRI image → real or fake probability.

    Architecture: Conv blocks with progressive downsampling → Dense → sigmoid
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, 4, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="MRI_Discriminator")


class BrainMRIGAN:
    """
    DCGAN for brain MRI synthesis.

    Train on real MRI images of a specific tumor class to generate
    additional synthetic training samples.
    """

    def __init__(self, latent_dim=128, img_size=240):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.generator = build_generator(latent_dim, img_size)
        self.discriminator = build_discriminator(img_size)

        self.g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated, training=True)

            # Label smoothing for stability
            real_labels = tf.ones_like(real_output) * 0.9
            fake_labels = tf.zeros_like(fake_output)

            d_loss_real = self.loss_fn(real_labels, real_output)
            d_loss_fake = self.loss_fn(fake_labels, fake_output)
            d_loss = d_loss_real + d_loss_fake

            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def train(self, dataset, epochs=100, log_every=10):
        """Train the GAN on real MRI images."""
        for epoch in range(epochs):
            for batch in dataset:
                losses = self.train_step(batch)

            if (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}/{epochs} — D Loss: {losses['d_loss']:.4f}, G Loss: {losses['g_loss']:.4f}")

        self.generator.save("mri_generator.keras")
        print("Generator saved: mri_generator.keras")

    def generate_samples(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic MRI images."""
        noise = tf.random.normal([n_samples, self.latent_dim])
        generated = self.generator(noise, training=False).numpy()
        # Scale from [-1, 1] to [0, 255]
        generated = ((generated + 1) * 127.5).astype(np.uint8)
        return generated

    def augment_dataset(self, real_images: np.ndarray, augment_ratio: float = 0.5) -> np.ndarray:
        """
        Augment real dataset with synthetic images.

        augment_ratio: fraction of synthetic images to add (0.5 = add 50% more)
        """
        n_synthetic = int(len(real_images) * augment_ratio)
        synthetic = self.generate_samples(n_synthetic)
        combined = np.concatenate([real_images, synthetic], axis=0)
        np.random.shuffle(combined)
        return combined
