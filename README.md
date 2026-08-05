# Breast Cancer Classification on BreakHis (Binary)

## Overview

This project performs **binary classification** (Benign vs Malignant) on the [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) histopathology dataset using a custom Convolutional Neural Network trained **from scratch**.

The primary focus of this work is **methodological correctness**, particularly avoiding patient-level data leakage.

## Dataset

- **Name**: BreakHis
- **Task**: Binary Classification (Benign vs Malignant)
- **Total Images**: ~7,909
- **Total Patients**: 82
- **Magnifications**: 40X, 100X, 200X, 400X (all magnifications used together)
- **Image Size**: Original ~700×460 → Resized to 224×224

## Key Methodology

### Patient-Level Split

To prevent data leakage, the dataset was split at the **patient level** (not image level).

- All images from the same patient are kept together in either Train, Validation, or Test set.
- Split ratio: **75% Train / 10% Validation / 15% Test**
- Stratified by class to preserve class distribution.

### Data Augmentation

Applied **only** on the training set:

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (90°)
- ColorJitter (brightness, contrast, saturation)
- Resize to 224×224
- ImageNet normalization

Validation and Test sets only receive resizing + normalization.

## Model Architecture

Custom CNN trained from scratch:

| Block | Layers                                      | Output Size |
|-------|---------------------------------------------|-------------|
| 1     | Conv 3→32 + BN + ReLU (×2) + MaxPool        | 112×112     |
| 2     | Conv 32→64 + BN + ReLU (×2) + MaxPool       | 56×56       |
| 3     | Conv 64→128 + BN + ReLU (×2) + MaxPool      | 28×28       |
| 4     | Conv 128→256 + BN + ReLU (×2) + MaxPool     | 14×14       |
| 5     | Conv 256→512 + BN + ReLU + AdaptiveAvgPool  | 1×1         |
| Head  | Dropout → Linear 512→128 → ReLU → Dropout → Linear 128→2 | - |

- **Total Trainable Parameters**: ~2.42 million

## Training Setup

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW` (lr = 1e-3, weight_decay = 1e-4)
- **Scheduler**: `ReduceLROnPlateau` (mode="max", monitor Validation AUC)
- **Epochs**: 25
- **Batch Size**: 32
- **Model Selection**: Best model chosen by highest Validation AUC

## Results

### Best Validation Performance
- **Best Epoch**: 17
- **Validation AUC**: **0.9977**

### Final Test Set Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 81.13%  |
| **AUC**    | **0.9362** |

**Classification Report (Test Set)**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Benign      | 0.64      | 0.96   | 0.77     | 403     |
| Malignant   | 0.98      | 0.74   | 0.84     | 853     |

**Confusion Matrix**

```
                  Predicted
                Benign    Malignant
Actual Benign     387         16
Actual Malignant  221        632
```

### Interpretation

- Strong ranking ability (AUC = 0.936)
- High precision on Malignant (0.98) — when the model predicts cancer, it is usually correct
- Lower recall on Malignant (0.74) — the model misses some cancer cases (conservative behaviour)

## Project Structure

```
├── Data/
│   ├── benign/
│   └── malignant/
├── best_model.pth          # Best model weights (by Val AUC)
├── main.ipynb              # Full training & evaluation notebook
└── README.md
```

## How to Run

1. Place the BreakHis dataset in the `Data/` folder with the structure shown above.
2. Open `main.ipynb` and run all cells sequentially.
3. The best model will be saved as `best_model.pth`.

## Notes & Future Work

- This model was trained **entirely from scratch** (no transfer learning).
- The main priority was correct experimental design (patient-level split) rather than chasing inflated accuracy through leakage.
- Possible next steps:
  - Transfer Learning (ResNet50, EfficientNet, ConvNeXt, etc.)
  - Class weights or focal loss to improve Malignant recall
  - Patient-level evaluation (majority vote per patient)
  - Threshold tuning on the validation set

## Author

Mackhem Chuah
```
