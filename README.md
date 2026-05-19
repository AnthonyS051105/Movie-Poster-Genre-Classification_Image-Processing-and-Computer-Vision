# Movie Poster Genre Classification

**Course:** Image Processing and Computer Vision – Even Semester 2025/2026  
**University:** Universitas Gadjah Mada  
**Lecturer:** Dr.Eng. Ir. Sunu Wibirama, S.T., M.Eng., IPM.

---

## Group Members

| Name | Student ID |
|------|------------|
| Raditya Ryan Narotama | 23/518350/TK/57045 |
| Yohanes Anthony Saputra | 24/536237/TK/59524 |
| Shafiyah Nuril Hayya | 24/540586/TK/60019 |

---

## Project Overview

This project addresses the task of predicting movie genres from poster images using deep learning. Each poster is a multi-label classification problem — a single poster can belong to one or more of **14 genres** simultaneously, as labeled from IMDb data.

We compare three pretrained CNN architectures:

| Model | Notebook |
|-------|----------|
| DenseNet121 | [densenet121.ipynb](densenet121.ipynb) |
| EfficientNet-B0 | [efficientnet-b0.ipynb](efficientnet-b0.ipynb) |
| ResNet50 | [resnet50.ipynb](resnet50.ipynb) |

### Genres

`action` · `adventure` · `animation` · `comedy` · `crime` · `drama` · `family` · `fantasy` · `horror` · `musical` · `mystery` · `romance` · `scifi` · `thriller`

---

## Dataset

| Split | Size | Description |
|-------|------|-------------|
| Train | 192 images | Used with WeightedRandomSampler |
| Validation | 50 images | Used for best model selection |
| Test | 50 images | Final evaluation |
| **Total** | **292 images** | Movie poster images as tensor |

- **Source:** Kaggle — `shafiyahnurilhayya/movie-genre-preprocessed`  
- **Original dataset:** Provided by the course via Google Drive (`genre.xlsx` + `images/train` + `images/test`)
- **Format:** Preprocessed `.pt` file, normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), input size 224×224
- **Label format:** Multi-hot binary vector of length 14
- **Class imbalance:** Minority genres (musical, horror, romance, family) have very few samples — addressed with `pos_weight` and `WeightedRandomSampler`

---

## Methodology

### Training Strategy

All three models follow a **two-phase transfer learning** approach:

- **Phase 1 — Classifier Head Training**: The backbone is frozen; only the new classification head is trained at `lr=1e-3`
- **Phase 2 — Full Fine-tuning**: All parameters are unfrozen and trained at `lr=1e-5` to preserve pretrained features while adapting to the domain

### Model Configurations

| Configuration | DenseNet121 | EfficientNet-B0 | ResNet50 |
|---------------|-------------|-----------------|----------|
| Phase 1 Epochs | 10 | 15 | 30 |
| Phase 2 Epochs | 40 | 35 | 20 |
| Total Epochs | 50 | 50 | 50 |
| Loss Function | FocalLoss (γ=2.0) | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Optimizer | Adam | Adam | Adam |
| LR Scheduler | CosineAnnealingLR | CosineAnnealingLR | StepLR |
| Dropout | 0.2 | 0.3 | — |
| Prediction Threshold | 0.5 | 0.5 | 0.5 |
| Best Model Criterion | Val F1-macro | Val F1-macro | Val F1-macro |

### Class Imbalance Handling

- **DenseNet121**: Manually boosted `pos_weight` for problematic genres (romance ×3, musical ×3, horror ×2, family ×1.5) combined with **Focal Loss** (γ=2.0) to focus training on frequently misclassified genres
- **EfficientNet-B0**: Standard `pos_weight` with threshold **0.5**, same as the other models for fair comparison
- **ResNet50**: Standard `pos_weight`; Phase 2 selectively unfreezes only `layer4` + `fc` for targeted fine-tuning

---

## Results

### Model Comparison (Test Set)

| Metric | DenseNet121 | EfficientNet-B0 | ResNet50 |
|--------|-------------|-----------------|----------|
| **F1-score (Micro)** | 0.4872 | 0.4519 | **0.5085** |
| **F1-score (Macro)** | **0.4143** | 0.3795 | 0.4178 |
| **F1-score (Weighted)** | 0.5539 | 0.5302 | **0.5571** |
| **mAP (Macro)** | **0.4741** | 0.4485 | — |
| **AUC-ROC (Weighted)** | 0.6528 | 0.6420 | **0.6916** |
| **Hamming Loss** | 0.3429 | 0.3743 | **0.2900** |
| **Exact Match Accuracy** | 0.0000 | 0.0000 | 0.0000 |
| **Total Parameters** | 6,968,206 | **4,025,482** | 23,536,718 |
| **Inference Time / Image** | 14.41 ms | 7.56 ms | **2.73 ms** |
| **Training Time / Epoch** | 1.9 s | **0.8 s** | ~2–3 s |

> **Note on Exact Match Accuracy = 0:** This is expected behavior for multi-label classification on a small dataset with overlapping genres. This strict metric requires every single genre label to be correct simultaneously.

### Per-Genre Performance — DenseNet121 (threshold=0.5)

| Genre | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| action | 0.7353 | 0.6944 | 0.7143 | 36 |
| adventure | 0.6000 | 0.6818 | 0.6383 | 22 |
| animation | 0.6154 | 1.0000 | 0.7619 | 8 |
| comedy | 0.4286 | 0.4615 | 0.4444 | 13 |
| crime | 0.4231 | 0.8462 | 0.5641 | 13 |
| drama | 0.7407 | 0.5882 | 0.6557 | 34 |
| family | 0.2500 | 0.7500 | 0.3750 | 4 |
| fantasy | 0.1538 | 0.1818 | 0.1667 | 11 |
| horror | 0.1053 | 0.6667 | 0.1818 | 3 ⚠ |
| musical | 0.0000 | 0.0000 | 0.0000 | 0 ⚠ |
| mystery | 0.3000 | 0.4615 | 0.3636 | 13 |
| romance | 0.0000 | 0.0000 | 0.0000 | 1 ⚠ |
| scifi | 0.3846 | 0.3571 | 0.3704 | 14 |
| thriller | 0.6111 | 0.5238 | 0.5641 | 21 |
| **Micro avg** | **0.4145** | **0.5907** | **0.4872** | 193 |
| **Macro avg** | **0.3820** | **0.5152** | **0.4143** | — |
| **Weighted avg** | **0.5491** | **0.5907** | **0.5539** | — |

### Per-Genre Performance — EfficientNet-B0 (threshold=0.5)

| Genre | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| action | 0.7083 | 0.4722 | 0.5667 | 36 |
| adventure | 0.6000 | 0.5455 | 0.5714 | 22 |
| animation | 0.6154 | 1.0000 | 0.7619 | 8 |
| comedy | 0.3889 | 0.5385 | 0.4516 | 13 |
| crime | 0.3750 | 0.6923 | 0.4865 | 13 |
| drama | 0.8235 | 0.8235 | 0.8235 | 34 |
| family | 0.2143 | 0.7500 | 0.3333 | 4 ⚠ |
| fantasy | 0.2308 | 0.2727 | 0.2500 | 11 |
| horror | 0.0000 | 0.0000 | 0.0000 | 3 ⚠ |
| musical | 0.0000 | 0.0000 | 0.0000 | 0 ⚠ |
| mystery | 0.1818 | 0.3077 | 0.2286 | 13 |
| romance | 0.0000 | 0.0000 | 0.0000 | 1 ⚠ |
| scifi | 0.2500 | 0.2143 | 0.2308 | 14 |
| thriller | 0.5600 | 0.6667 | 0.6087 | 21 |
| **Micro avg** | **0.3789** | **0.5596** | **0.4519** | 193 |
| **Macro avg** | **0.3534** | **0.4488** | **0.3795** | — |
| **Weighted avg** | **0.5315** | **0.5596** | **0.5302** | — |

### Per-Genre Performance — ResNet50 (threshold=0.5)

| Genre | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| action | 0.8000 | 0.5556 | 0.6557 | 36 |
| adventure | 0.6154 | 0.7273 | 0.6667 | 22 |
| animation | 0.5000 | 1.0000 | 0.6667 | 8 |
| comedy | 0.6364 | 0.5385 | 0.5833 | 13 |
| crime | 0.4615 | 0.4615 | 0.4615 | 13 |
| drama | 0.8077 | 0.6176 | 0.7000 | 34 |
| family | 0.2143 | 0.7500 | 0.3333 | 4 |
| fantasy | 0.2667 | 0.3636 | 0.3077 | 11 |
| horror | 0.1111 | 0.3333 | 0.1667 | 3 ⚠ |
| musical | 0.0000 | 0.0000 | 0.0000 | 0 ⚠ |
| mystery | 0.4667 | 0.5385 | 0.5000 | 13 |
| romance | 0.0000 | 0.0000 | 0.0000 | 1 ⚠ |
| scifi | 0.3333 | 0.2857 | 0.3077 | 14 |
| thriller | 0.7273 | 0.3810 | 0.5000 | 21 |
| **Micro avg** | **0.4773** | **0.5441** | **0.5085** | 193 |
| **Macro avg** | **0.4243** | **0.4680** | **0.4178** | — |
| **Weighted avg** | **0.6124** | **0.5441** | **0.5571** | — |

> ⚠ = minority genre (support ≤ 3 in test set)

---

## Project Structure

```
Movie-Poster-Genre-Classification/
├── densenet121.ipynb       # DenseNet121 training & evaluation
├── efficientnet-b0.ipynb   # EfficientNet-B0 training & evaluation
├── resnet50.ipynb          # ResNet50 training & evaluation
└── README.md
```

---

## How to Run

> All notebooks are designed to run on **Kaggle** with a Tesla T4 GPU.

**Requirements:**
```
torch >= 2.0
torchvision >= 0.15
scikit-learn
matplotlib
numpy
pandas
```

**Steps:**
1. Upload the notebook to Kaggle
2. Add the dataset `shafiyahnurilhayya/movie-genre-preprocessed` as input
3. Enable GPU (Settings → Accelerator → GPU T4)
4. Click **Run All** — all cells execute automatically top to bottom
5. Outputs are saved to `/kaggle/working/`

**Consistent inference across models:**  
All notebooks use `sample_indices.npy` with `seed=42` so the 10 visualized test samples are **identical** across all three models, enabling fair visual comparison.

---

## Experimental Environment

| Component | Details |
|-----------|---------|
| GPU | Tesla T4 (15.6 GB VRAM) |
| PyTorch | 2.10.0+cu128 |
| Torchvision | 0.25.0+cu128 |
| Python | 3.12 |
| Platform | Kaggle Notebooks |

---

## Key Findings

- **ResNet50** achieves the best AUC-ROC (0.6916) and lowest Hamming Loss (0.2900), indicating the strongest overall class discrimination despite being the largest model (23.5M parameters)
- **EfficientNet-B0** is the most efficient model: fewest parameters (4M) and fastest inference (7.56 ms), but scores the lowest across most metrics (F1-macro 0.3795, AUC-ROC 0.6420) when using a uniform threshold of 0.5
- **DenseNet121** achieves the highest mAP (0.4741) and a competitive F1-macro (0.4143), reflecting better-calibrated probability scores enabled by Focal Loss combined with manually boosted `pos_weight`
- All models fail on **extreme minority genres** (musical, romance) with 0–1 test samples — a fundamental limitation of the small dataset size (292 total images)
- With all three models now using the same threshold of **0.5**, ResNet50 leads in F1-micro (0.5085), AUC-ROC (0.6916), and Hamming Loss (0.2900), making it the strongest overall performer
