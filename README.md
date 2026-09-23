# 🌿 Crop Disease Detection

A Convolutional Neural Network (CNN) that classifies tomato, potato, and
bell pepper leaves as healthy or diseased from a photo, trained on the
[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset.

## Motivation

Crop disease is a major cause of yield loss, and by the time symptoms are
obvious to the naked eye, it's often too late to save the plant. The goal
here is a simple proof of concept: a model that takes a phone photo of a
leaf and identifies the disease early enough for a farmer to act on it.

## Dataset

- **Source:** [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) (via `kagglehub`)
- **Content:** Leaf images across tomato, potato, and bell pepper, both healthy and affected by various diseases
- **Split:** 80% train / 20% validation

### Manual dataset download (if `kagglehub` fails)

1. Go to the [dataset page](https://www.kaggle.com/datasets/emmarex/plantdisease) and click Download (requires a free Kaggle account).
2. Unzip it so you have a `PlantVillage/` folder with one subfolder per class.
3. Place that folder in the project root (same level as `crop_disease_detection.py`).

## Model Architecture

A compact CNN — nothing exotic, on purpose, since the goal was to learn
the fundamentals of image classification before reaching for transfer
learning:

```
Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool
-> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(num_classes, softmax)
```

- **Input size:** 128x128x3
- **Augmentation:** rotation, zoom, horizontal flip (helps generalize to real-world photos taken at odd angles)
- **Optimizer:** Adam | **Loss:** categorical cross-entropy
- **Learning type:** Supervised — every image is labeled with its disease/healthy class

## Setup

```bash
git clone https://github.com/NinjaVinja/crop-disease-detection.git
cd crop-disease-detection
pip install -r requirements.txt
python crop_disease_detection.py
```

A GPU is recommended (Colab's free GPU tier works well) — training on CPU
is possible but noticeably slower for 15 epochs over ~20k images.

### Running on Google Colab

1. Upload `crop_disease_detection.py` to Colab, or paste its contents into a notebook cell.
2. Go to **Runtime → Change runtime type → GPU**.
3. Run the script (`!python crop_disease_detection.py` in a cell, or just execute the pasted cells).
   - `kagglehub` is auto-installed if it's missing — no manual `!pip install` needed.
   - At the end of the run, a file-upload picker will pop up automatically so you can test the model on your own leaf photo right away.

## What the script does

1. Downloads the dataset and builds train/validation generators
2. Shows a quick sample grid of augmented images (sanity check)
3. Builds and summarizes the CNN
4. Evaluates the untrained model as a **baseline** (random-guess accuracy)
5. Trains for 15 epochs
6. Evaluates again post-training and reports the improvement over baseline
7. Logs every run's numbers to `training_log.csv` and plots accuracy across all runs, so repeated experiments are comparable rather than overwriting each other
8. Plots per-epoch accuracy/loss curves for the current run
9. Saves the trained model to `crop_disease_model.h5`

To test the model on your own leaf photo, call `predict_image()` with a
path to a local image (see the commented-out line at the bottom of `main()`).

## Results

Baseline (untrained) accuracy sits near random guessing, and validation
accuracy climbs substantially after 15 epochs of training — exact numbers
depend on the run and are logged in `training_log.csv`. Re-running the
script appends a new row, so accuracy trends across experiments are easy
to track over time.

## Future Improvements

- Swap the from-scratch CNN for **transfer learning** (MobileNet/ResNet
  pretrained on ImageNet) — should improve accuracy with less training time
- Expand beyond tomato/potato/pepper to more crop types
- Package the trained model behind a simple mobile-friendly interface so
  a farmer could snap a photo and get an instant result in the field
- Explore techniques for handling images taken in inconsistent real-world
  lighting/backgrounds, since PlantVillage images are mostly clean lab shots

## Project Context

Built as a course project for **Artificial Intelligence** (6th semester,
BSCS) at the University of Central Punjab, under the supervision of
**Prof. Tajjamul**.

## Author

**Muhammad Taha Ahmad** ([@NinjaVinja](https://github.com/NinjaVinja))
