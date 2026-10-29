# Corruption-Aware Traffic Sign Classification Using Swin Transformer

## 1. Project Overview

This project implements traffic sign classification using the German Traffic Sign Recognition Benchmark (GTSRB). It compares a fine-tuned Swin-Tiny baseline with a proposed corruption-aware Swin Transformer designed to improve classification under image corruptions.

## 2. Dataset

- Dataset: GTSRB
- Number of classes: 43
- Training images: 39,209
- Official test images: 12,630
- Input resolution: 224 × 224 pixels

Images are converted to RGB, resized, and normalized using ImageNet mean and standard deviation.

## 3. Methodology

### Baseline Model

A Swin-Tiny Transformer extracts image features and classifies traffic signs into 43 classes. The baseline is trained and fine-tuned using cross-entropy loss and AdamW optimization.

### Proposed Corruption-Aware Model

The proposed model adds two auxiliary prediction heads to Swin-Tiny:

- Corruption-type head: 15 categories, including clean images and 14 corruption types.
- Severity head: 5 severity levels.

The predicted condition probabilities are passed to a feature router. The router produces feature modulation parameters, gamma and beta, which adjust the extracted features before final classification.

The total training loss is:

L = L_class + lambda_c * L_corruption + lambda_s * L_severity

Severity loss is ignored for clean samples.

### Corruption Evaluation

The model is evaluated across 14 corruption types and 5 severity levels, giving 70 corruption conditions.

## 4. Current Results

| Metric | Fine-tuned Swin | Corruption-aware Swin |
|---|---:|---:|
| Clean official test accuracy | 98.86% | 99.27% |
| Mean corrupted accuracy | 68.58% | 88.71% |

The observed mean corrupted-accuracy improvement is 20.13 percentage points in the seed-controlled comparison.

These figures should be interpreted in light of the evaluation protocol and reproducibility limitations.

## 5. Main Code Files

- `datasets/gtsrb_dataset.py`: loads GTSRB images and labels.
- `datasets/corruption_aware_dataset.py`: creates corruption-augmented training samples.
- `models/swin_baseline.py`: baseline architecture.
- `models/train_baseline.py`: trains the baseline.
- `models/train_swin_finetune.py`: fine-tunes the baseline.
- `models/corruption_aware_swin.py`: proposed architecture.
- `models/train_corruption_aware.py`: trains the proposed model.
- `models/evaluate_corruption_aware.py`: evaluates clean test accuracy.
- `models/evaluate_corruption_aware_corruptions.py`: evaluates corruption robustness.
- `models/controlled_corruption_comparison.py`: compares the two models.
- `models/summarize_controlled_results.py`: summarizes results and creates plots.

## 6. Running the Project

Activate the project's Python virtual environment and install the dependencies listed in `requirements.txt`.

Run the training and evaluation scripts from the project root. Pretrained or trained checkpoints must be available in the `checkpoints/` directory.

## 7. Limitations and Future Work

Future work includes an ablation study of the auxiliary heads and feature router, stricter reproducibility using saved corrupted images, and evaluation on additional traffic sign datasets if required.
