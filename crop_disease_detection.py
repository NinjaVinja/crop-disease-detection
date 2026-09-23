"""
Crop Disease Detection using a Convolutional Neural Network (CNN)
-------------------------------------------------------------------
Trains an image classifier on the PlantVillage dataset to identify
diseased vs. healthy leaves across tomato, potato, and bell pepper
crops.

Author: Muhammad Taha Ahmad (NinjaVinja)
Course: Artificial Intelligence, Semester 6
Supervision: Prof. Tajjamul

Usage:
    python crop_disease_detection.py

A GPU is strongly recommended (Colab's free tier works fine) since
training a 3-layer CNN on ~20k images is otherwise slow.
"""

import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def in_colab() -> bool:
    """True when running inside a Google Colab notebook."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_kagglehub_installed():
    """
    Installs kagglehub if it's missing. Colab doesn't ship it by
    default, so this saves a manual `!pip install` cell.
    """
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        print("kagglehub not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])

# ---------------------------------------------------------------------------
# Config — tweak these if you want to experiment without touching the logic
# ---------------------------------------------------------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
MODEL_OUT_PATH = "crop_disease_model.h5"
LOG_FILE = "training_log.csv"


def get_dataset_dir() -> str:
    """
    Downloads the PlantVillage dataset via kagglehub and returns the
    path to the folder containing one subfolder per class.

    Falls back to a manually-downloaded copy at ./PlantVillage if
    kagglehub isn't available or Kaggle auth isn't set up — see the
    README for the manual download steps.
    """
    try:
        ensure_kagglehub_installed()
        import kagglehub
        path = kagglehub.dataset_download("emmarex/plantdisease")
        dataset_dir = os.path.join(path, "PlantVillage")
    except Exception as err:
        print(f"kagglehub download failed ({err}); falling back to local ./PlantVillage")
        dataset_dir = "PlantVillage"

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Could not find dataset at {dataset_dir}. "
            "Download it manually from Kaggle and place it in this folder — "
            "see the README's 'Manual dataset download' section."
        )
    return dataset_dir


def build_data_generators(dataset_dir: str):
    """
    Builds the train/validation generators with normalization and
    light augmentation (rotation, zoom, horizontal flip) so the model
    generalizes better on unseen leaf photos.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    return train_data, val_data


def preview_samples(train_data, class_names, n=6):
    """Sanity-check a batch of augmented training images before training."""
    images, labels = next(train_data)
    plt.figure(figsize=(12, 8))
    for i in range(n):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def build_model(num_classes: int) -> Sequential:
    """
    A compact 3-block CNN. Nothing exotic — three Conv+Pool blocks to
    pull out edges/color/texture features, then a dense head with
    dropout to keep it from memorizing the training set.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def log_run(baseline_acc, final_train_acc, final_val_acc, final_val_loss):
    """
    Appends this run's numbers to a CSV so accuracy can be compared
    across multiple training attempts (different epoch counts, tweaks,
    etc.) instead of only ever seeing the latest run.
    """
    record = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": EPOCHS,
        "baseline_accuracy_%": round(baseline_acc * 100, 2),
        "final_train_accuracy_%": round(final_train_acc * 100, 2),
        "final_val_accuracy_%": round(final_val_acc * 100, 2),
        "final_val_loss": round(final_val_loss, 4),
    }

    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)
    else:
        log_df = pd.DataFrame([record])

    log_df.to_csv(LOG_FILE, index=False)
    return log_df


def plot_run_history(log_df):
    """Line chart comparing baseline vs. post-training accuracy across all logged runs."""
    plt.figure(figsize=(10, 5))
    run_numbers = range(1, len(log_df) + 1)
    plt.plot(run_numbers, log_df["baseline_accuracy_%"], marker="x", linestyle="--", label="Baseline (before training)")
    plt.plot(run_numbers, log_df["final_val_accuracy_%"], marker="o", label="Validation accuracy (after training)")
    plt.xlabel("Training run #")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy across all training runs")
    plt.xticks(list(run_numbers))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_training_curves(history):
    """Standard accuracy/loss-per-epoch curves for this specific run."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_image(model, image_path, class_names):
    """Runs the trained model on a single leaf photo and shows the prediction."""
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.show()

    return predicted_class, confidence


def predict_uploaded_image(model, class_names):
    """
    Colab-only helper: opens a file-picker so you can upload a leaf
    photo from your computer and immediately see the prediction.
    No-ops with a message if not running in Colab.
    """
    if not in_colab():
        print("predict_uploaded_image() only works inside Google Colab.")
        print("Locally, call predict_image(model, 'path/to/image.jpg', class_names) instead.")
        return

    from google.colab import files
    uploaded = files.upload()
    for filename in uploaded.keys():
        predict_image(model, filename, class_names)


def main():
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    dataset_dir = get_dataset_dir()
    train_data, val_data = build_data_generators(dataset_dir)

    num_classes = train_data.num_classes
    class_names = list(train_data.class_indices.keys())
    print("Total classes (diseases):", num_classes)
    print("Class names:", class_names)

    preview_samples(train_data, class_names)

    model = build_model(num_classes)
    model.summary()

    print("\n=== Baseline result (before training) ===")
    baseline_loss, baseline_acc = model.evaluate(val_data)
    print(f"Before training -> Loss: {baseline_loss:.4f} | Accuracy: {baseline_acc * 100:.2f}%")

    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    print("\n=== Result (after training) ===")
    final_loss, final_acc = model.evaluate(val_data)
    print(f"After training -> Loss: {final_loss:.4f} | Accuracy: {final_acc * 100:.2f}%")
    print(f"Improvement over baseline: {(final_acc - baseline_acc) * 100:.2f}%")

    log_df = log_run(baseline_acc, history.history["accuracy"][-1], final_acc, final_loss)
    print(f"\nLogged this run. Total runs so far: {len(log_df)}")
    plot_run_history(log_df)
    plot_training_curves(history)

    model.save(MODEL_OUT_PATH)
    print(f"Model saved to {MODEL_OUT_PATH}")

    # In Colab, this opens a file picker to upload and instantly test a leaf photo.
    # Comment it out if you just want to train without testing each run.
    predict_uploaded_image(model, class_names)


if __name__ == "__main__":
    main()
