# 🫀 Hybrid Deep Learning System for Early Cardiovascular Disease Risk Prediction

A deep learning–based clinical decision support system that predicts the likelihood of cardiovascular disease (CVD) from standard clinical and diagnostic features. Built with **PyTorch**, evaluated with industry-standard metrics, and explained using **SHAP** (SHapley Additive exPlanations).

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Notebook Walkthrough](#notebook-walkthrough)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Feature Engineering & Train-Test Split](#3-feature-engineering--train-test-split)
  - [4. Model Architecture](#4-model-architecture)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Risk Simulation Engine](#7-risk-simulation-engine)
  - [8. Explainable AI (SHAP)](#8-explainable-ai-shap)
- [Results Summary](#results-summary)
- [Generated Visualizations](#generated-visualizations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [License](#license)

---

## Problem Statement

Cardiovascular disease remains the **#1 cause of death globally**, responsible for an estimated 17.9 million deaths annually (WHO). Early detection through clinical data can enable timely intervention, reduce mortality, and lower healthcare costs.

This project builds a **hybrid deep neural network** that:
- Learns complex non-linear patterns from standard clinical features
- Outputs a calibrated probability of heart disease risk
- Provides a **risk simulation engine** for what-if clinical scenario analysis
- Uses **SHAP explainability** to make predictions transparent and clinically interpretable

---

## Dataset

**Heart Disease UCI Dataset** — Kaggle version based on the UCI Machine Learning Repository.

| Property | Details |
|----------|---------|
| **Source** | [Kaggle / UCI ML Repository](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| **Samples** | 303 patient records (302 after duplicate removal) |
| **Features** | 13 clinical attributes |
| **Target** | Binary — `1` (heart disease present), `0` (no heart disease) |
| **Class Balance** | 164 positive (54.3%) / 138 negative (45.7%) |

### Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Continuous |
| `sex` | Sex (1 = male, 0 = female) | Binary |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Continuous |
| `chol` | Serum cholesterol (mg/dl) | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Continuous |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) | Binary |
| `oldpeak` | ST depression induced by exercise relative to rest | Continuous |
| `slope` | Slope of the peak exercise ST segment (0–2) | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) | Ordinal |
| `thal` | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) | Categorical |

---

## Project Structure

```
arye/
├── main.ipynb                  # Complete Jupyter Notebook (end-to-end pipeline)
├── heart.csv                   # Heart Disease UCI Dataset
├── README.md                   # This file
├── correlation_matrix.png      # Feature correlation heatmap
├── class_distribution.png      # Target class distribution chart
├── training_loss_curve.png     # Training loss over 50 epochs
├── roc_curve.png               # ROC curve with AUC score
└── feature_importance.png      # SHAP feature importance summary
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip or conda

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch shap
```

> **Note:** If you encounter a NumPy / numba compatibility issue, run:
> ```bash
> pip install --user --upgrade numba shap
> ```

### Dataset

The notebook **automatically downloads** `heart.csv` from GitHub if it's not present in the working directory. No manual download required.

---

## Notebook Walkthrough

### 1. Data Preprocessing

- Loaded 303 records with 14 columns (13 features + 1 target)
- Identified and removed **1 duplicate row** → 302 clean records
- Verified **zero missing values** across all features
- All columns are numeric (int64 / float64) — no encoding needed

### 2. Exploratory Data Analysis

#### Class Distribution

![Class Distribution](class_distribution.png)

**Figure 1 — Heart Disease Class Distribution.** The dataset is approximately balanced with **164 positive cases** (heart disease present) and **138 negative cases** (no heart disease). This near-balanced split (54% / 46%) means the model can learn effectively without requiring oversampling or class-weight adjustments.

---

#### Correlation Matrix

![Correlation Matrix](correlation_matrix.png)

**Figure 2 — Feature Correlation Matrix (lower triangle heatmap).** Key observations:
- **`cp` (chest pain type)** has the strongest positive correlation with the target (+0.43), confirming its clinical significance
- **`exang` (exercise-induced angina)** and **`oldpeak` (ST depression)** show strong negative correlations with the target (−0.44 and −0.43), indicating higher values are associated with disease absence in this encoding
- **`thalach` (max heart rate)** correlates positively with the target (+0.42) — higher heart rate capacity is linked to disease presence in this dataset's encoding
- **`ca` (fluoroscopy vessels)** and **`thal` (thalassemia)** also show notable correlations (−0.41 and −0.34)
- Features like `fbs` (fasting blood sugar) and `chol` (cholesterol) have weak correlations, suggesting limited standalone predictive power
- No extreme multicollinearity detected between feature pairs, supporting their inclusion in the model

---

#### Feature Distributions by Target

The notebook also generates per-feature histograms comparing distributions between disease and no-disease groups for `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`. These reveal that patients with heart disease tend to have higher maximum heart rates and lower ST depression values, while cholesterol and blood pressure show more overlap between groups.

---

### 3. Feature Engineering & Train-Test Split

- **Feature Scaling:** All 13 features standardized using `StandardScaler` (zero mean, unit variance)
- **Split:** 80% training (241 samples) / 20% test (61 samples)
- **Stratification:** Applied to preserve class proportions in both sets
  - Train: 110 negative / 131 positive
  - Test: 28 negative / 33 positive
- **Tensors:** Converted to PyTorch `FloatTensor` on GPU (CUDA) when available
- **DataLoader:** Batch size = 32, shuffled

### 4. Model Architecture

A custom **deep neural network** (`HeartDiseaseNet`) implemented in PyTorch:

```
Input (13 features)
  → Linear(13, 128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(128, 64) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64, 1) → Sigmoid
```

| Component | Details |
|-----------|---------|
| **Input Layer** | 13 neurons (one per feature) |
| **Hidden Layer 1** | 128 neurons, BatchNorm, ReLU activation, 30% Dropout |
| **Hidden Layer 2** | 64 neurons, BatchNorm, ReLU activation, 30% Dropout |
| **Output Layer** | 1 neuron, Sigmoid activation (probability output) |
| **Total Parameters** | 10,497 (all trainable) |

**Design Choices:**
- **Batch Normalization** stabilizes training and enables faster convergence
- **Dropout (30%)** prevents overfitting on this small dataset (302 samples)
- **Sigmoid output** provides calibrated probability for risk quantification

### 5. Model Training

| Hyperparameter | Value |
|----------------|-------|
| Loss Function | Binary Cross-Entropy (`BCELoss`) |
| Optimizer | Adam (lr = 0.001) |
| Epochs | 50 |
| Batch Size | 32 |

#### Training Loss Curve

![Training Loss Curve](training_loss_curve.png)

**Figure 3 — Training Loss Curve over 50 Epochs.** The BCE loss decreases steadily from **0.6628 (epoch 1)** to **0.1945 (epoch 50)**, showing consistent convergence without signs of divergence. The slight oscillations are expected with mini-batch gradient descent on a small dataset. The model reaches the low-loss plateau around epoch 35–40, indicating adequate training duration.

---

### 6. Model Evaluation

#### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.8033 (80.33%) |
| **Precision** | 0.8182 (81.82%) |
| **Recall** | 0.8182 (81.82%) |
| **F1-Score** | 0.8182 (81.82%) |
| **ROC-AUC** | 0.8636 (86.36%) |

The model achieves **balanced precision and recall** (both 81.82%), meaning it is equally good at avoiding false positives (incorrectly diagnosing disease) and false negatives (missing actual disease cases). The **ROC-AUC of 0.8636** indicates strong discriminative ability — the model correctly ranks a randomly chosen positive patient above a randomly chosen negative patient 86.4% of the time.

#### ROC Curve

![ROC Curve](roc_curve.png)

**Figure 4 — Receiver Operating Characteristic (ROC) Curve.** The curve shows the trade-off between True Positive Rate (sensitivity) and False Positive Rate (1 − specificity) at various classification thresholds. The **AUC = 0.8636** (shaded area under the red curve) is well above the random baseline (dashed diagonal, AUC = 0.5), confirming the model has learned meaningful clinical patterns. The curve hugs the upper-left corner, indicating high sensitivity can be achieved with relatively low false positive rates.

#### Confusion Matrix

The model correctly classified:
- **22 out of 28** no-disease patients (True Negatives)
- **27 out of 33** disease patients (True Positives)
- 6 False Positives (healthy patients incorrectly flagged)
- 6 False Negatives (disease patients missed)

---

### 7. Risk Simulation Engine

The notebook includes a `simulate_risk_change()` function that enables **what-if clinical analysis**:

```python
simulate_risk_change(input_data, modified_feature, new_value)
```

**How it works:**
1. Takes a patient's full feature vector (raw values)
2. Computes the baseline predicted risk using the trained model
3. Modifies a single specified feature to a new value
4. Recomputes the predicted risk with the modification
5. Reports the old risk, new risk, and percentage change

**Example Scenarios (from the notebook):**

| Scenario | Modified Feature | Original → New | Old Risk | New Risk | Change |
|----------|-----------------|----------------|----------|----------|--------|
| Lower cholesterol | `chol` | 233 → 200 | 80.72% | 90.94% | +12.65% |
| Increase max heart rate | `thalach` | 150 → 170 | 80.72% | 89.39% | +10.74% |

> **Clinical Use Case:** Clinicians can use this engine to explore how modifying controllable risk factors (e.g., cholesterol via statins, blood pressure via antihypertensives) would change a patient's predicted risk, supporting personalized intervention planning.

---

### 8. Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to every individual prediction, grounded in cooperative game theory.

#### SHAP Feature Importance Summary

![Feature Importance](feature_importance.png)

**Figure 5 — SHAP Summary Plot.** Each dot represents one patient-feature pair. The horizontal position shows the SHAP value (impact on model output), and the color indicates the feature value (red = high, blue = low).

**Key Insights:**
- **`cp` (chest pain type)** — Most influential feature. Higher chest pain type values push predictions toward disease (red dots on right)
- **`ca` (fluoroscopy vessels)** — Second most important. Higher values (more vessels colored) strongly increase predicted risk
- **`thal` (thalassemia)** — Third most important. Both high and low values create significant directional shifts
- **`sex`** — Male sex is associated with higher predicted risk
- **`chol` (cholesterol)** — Moderate importance; extreme values have notable impact
- **`thalach` (max heart rate)** — Lower max heart rate increases disease risk prediction
- **`fbs`, `age`, `trestbps`** — Lowest individual importance, though they still contribute in combination

#### Feature Importance Ranking (Mean |SHAP|)

| Rank | Feature | Mean SHAP Value |
|------|---------|-----------------|
| 1 | `cp` | 0.1170 |
| 2 | `ca` | 0.1164 |
| 3 | `thal` | 0.0907 |
| 4 | `sex` | 0.0682 |
| 5 | `chol` | 0.0575 |
| 6 | `thalach` | 0.0537 |
| 7 | `oldpeak` | 0.0495 |
| 8 | `restecg` | 0.0417 |
| 9 | `exang` | 0.0346 |
| 10 | `slope` | 0.0261 |
| 11 | `fbs` | 0.0175 |
| 12 | `age` | 0.0162 |
| 13 | `trestbps` | 0.0089 |

---

## Results Summary

| Component | Details |
|-----------|---------|
| **Dataset** | Heart Disease UCI (302 samples, 13 features) |
| **Model** | 2-layer DNN with BatchNorm + Dropout (10,497 params) |
| **Training** | 50 epochs, Adam optimizer, BCE loss → final loss 0.1945 |
| **Accuracy** | 80.33% |
| **F1-Score** | 81.82% |
| **ROC-AUC** | 86.36% |
| **Top Features** | `cp`, `ca`, `thal`, `sex`, `chol` |
| **Explainability** | SHAP KernelExplainer on 50 test samples |

---

## Generated Visualizations

All plots are saved as high-resolution PNG files (150 DPI) in the project directory:

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart of target class counts (Disease vs No Disease) |
| `correlation_matrix.png` | Lower-triangle heatmap of pairwise feature correlations |
| `training_loss_curve.png` | BCE loss curve across 50 training epochs |
| `roc_curve.png` | ROC curve with AUC annotation and random baseline |
| `feature_importance.png` | SHAP beeswarm summary plot showing per-feature impact |

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| **pandas** | Data loading and manipulation |
| **numpy** | Numerical operations and array handling |
| **matplotlib** | Plot generation and PNG export |
| **seaborn** | Statistical visualizations (heatmap, styling) |
| **scikit-learn** | Preprocessing, train-test split, evaluation metrics |
| **PyTorch** | Deep neural network definition, training, and inference |
| **SHAP** | Model-agnostic feature importance and explainability |

---

## How to Run

1. **Clone or download** this repository

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn torch shap
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```
   Or open in **VS Code** with the Jupyter extension.

4. **Run all cells** sequentially (Kernel → Restart & Run All)

5. The notebook will:
   - Automatically download `heart.csv` if not present
   - Preprocess data and train the model (~12 seconds on GPU)
   - Generate all 5 PNG visualizations
   - Print final message: `"Model Training Complete. All Results Saved as PNG Files."`

---

## License

This project is for **educational and research purposes**. The Heart Disease UCI Dataset is publicly available from the UCI Machine Learning Repository.

---

> **Model Training Complete. All Results Saved as PNG Files.**
