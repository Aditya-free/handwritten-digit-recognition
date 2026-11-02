"""
train.py — Handwritten Digit Recognition (Track 2: Augmented & Deeper CNN)

Improvements over Track 1:
  - Real-world noise injection via tf.data pipeline
  - In-model data augmentation (RandomRotation, RandomZoom, RandomTranslation)
  - Deeper CNN architecture (3 conv blocks + BN + Dropout)
  - Adam optimizer with ReduceLROnPlateau and EarlyStopping callbacks
  - Manual validation split from training data
"""

import numpy as np
import tensorflow as tf

# ── 1. Load & preprocess MNIST ─────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = x_train[..., tf.newaxis]  # (60000, 28, 28, 1)
x_test  = x_test[..., tf.newaxis]   # (10000, 28, 28, 1)

# ── 2. Validation split ────────────────────────────────────────────────────────
# Split 10% of training data for validation before building the pipeline
val_size = int(0.1 * len(x_train))
x_val,   y_val   = x_train[:val_size], y_train[:val_size]
x_train, y_train = x_train[val_size:], y_train[val_size:]

# ── 3. Real-world noise injection ──────────────────────────────────────────────
# Digits captured from cameras, phone photos, or scanned notebooks come with
# Gaussian noise, lighting variation, and ruled lines.  Simulating this during
# training forces the model to learn digit structure rather than pixel values,
# improving robustness on real-world inputs.
def add_real_world_noise(images, labels):
    """Apply synthetic noise to a batch of greyscale digit images.

    Augmentations:
      • Gaussian noise  — simulates sensor/camera noise
      • Brightness jitter — simulates lighting variation
      • Random horizontal lines — simulates notebook ruled lines
    """
    # Gaussian noise
    images = images + tf.random.normal(
        shape=tf.shape(images), mean=0.0, stddev=0.05
    )

    # Random brightness shift (multiplicative)
    images = images * tf.random.uniform([], 0.8, 1.2)

    # Clip to valid pixel range
    images = tf.clip_by_value(images, 0.0, 1.0)

    # Random horizontal lines (simulating notebook lines) — 30% probability per image
    batch_size = tf.shape(images)[0]

    def draw_lines_on_image(image):
        """Draw 1-3 random horizontal lines on a single image."""
        # Decide whether to draw lines for this image (30% chance)
        draw = tf.random.uniform([], 0.0, 1.0) < 0.3

        def apply_lines(img):
            # Number of lines: 1 to 3
            num_lines = tf.random.uniform([], 1, 4, dtype=tf.int32)

            def add_one_line(i, current_img):
                y_pos = tf.random.uniform([], 0, 28, dtype=tf.int32)
                # Build a mask: zeros at y_pos row, ones elsewhere
                mask_top    = tf.ones([y_pos, 28, 1],      dtype=tf.float32)
                mask_line   = tf.zeros([1, 28, 1],         dtype=tf.float32)
                mask_bottom = tf.ones([28 - y_pos - 1, 28, 1], dtype=tf.float32)
                mask = tf.concat([mask_top, mask_line, mask_bottom], axis=0)
                return i + 1, current_img * mask

            _, lined_img = tf.while_loop(
                cond=lambda i, _: i < num_lines,
                body=add_one_line,
                loop_vars=[tf.constant(0), img],
            )
            return lined_img

        return tf.cond(draw, lambda: apply_lines(image), lambda: image)

    images = tf.map_fn(draw_lines_on_image, images, dtype=tf.float32)
    return images, labels


# ── 4. tf.data pipeline (shuffle → batch → noise → prefetch) ──────────────────
BATCH_SIZE = 64

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .map(add_real_world_noise, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# ── 5. Model architecture ──────────────────────────────────────────────────────
# Three convolutional blocks with BatchNormalization and Dropout,
# preceded by in-model augmentation layers (only active during training).
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # --- Augmentation (active during training, pass-through at inference) ---
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom((-0.1, 0.1)),
    tf.keras.layers.RandomTranslation(0.1, 0.1),

    # --- Conv Block 1: 32 filters ---
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # --- Conv Block 2: 64 filters ---
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # --- Conv Block 3: 128 filters ---
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # --- Classifier head ---
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax'),
], name="digit_cnn_v2")

model.summary()

# ── 6. Compile ─────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# ── 7. Callbacks ───────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── 8. Train ───────────────────────────────────────────────────────────────────
history = model.fit(
    train_dataset,
    epochs=25,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)

# ── 9. Evaluate on held-out test set ──────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Test loss     : {test_loss:.4f}")

# ── 10. Save model ─────────────────────────────────────────────────────────────
model.save("handwritten_digit_cnn.keras")
print("Model saved: handwritten_digit_cnn.keras")
