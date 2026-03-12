# TXGBoost: Triple Hybrid Model for Health State Prediction

A novel hybrid deep learning approach that combines temporal pattern learning with gradient boosting for Health state prediction.

## Overview

TXGBoost implements a **triple hybrid architecture** that significantly outperforms traditional feature engineering approaches by learning temporal patterns through a Time-Embedded RNN and combining them with handcrafted features for gradient boosting.

The framework now supports both **XGBoost** and **CatBoost** backends, with CatBoost achieving superior performance on AKI prediction tasks.

## Architecture

```
[Temporal Data] → Time-Embedded RNN (256-dim) → RNN Embeddings
                         ↓
[Static Features] → Global Statistics (mean/max/min/std/slope)
                         ↓
        [Last Values + Enhanced Static + RNN] → XGBoost/CatBoost
```

### Triple Feature Fusion

The model combines three complementary feature types:

1. **Last Values** (point-wise): Most recent observations for each temporal feature
2. **Enhanced Static Features**: Original static features + Global statistical summaries
3. **RNN Embeddings** (sequential): Learned temporal patterns from Time-Embedded RNN

## Performance Comparison

All results are averaged over 5-fold cross-validation on the same dataset:

### XGBoost Results

| Model | AUC | AUC-PR | vs Baseline AUC | vs Baseline AUC-PR |
|-------|-----|--------|-----------------|-------------------|
| **Baseline XGBoost** | 0.8562 ± 0.0090 | 0.6584 ± 0.0174 | - | - |
| **TBoostv3 (XGBoost)** | 0.8624 ± 0.0050 | 0.6678 ± 0.0097 | **+0.72%** | **+1.43%** |

### CatBoost Results

| Model | AUC | AUC-PR | vs Baseline AUC | vs Baseline AUC-PR |
|-------|-----|--------|-----------------|-------------------|
| **Baseline CatBoost** | 0.8636 ± 0.0074 | 0.6665 ± 0.0141 | - | - |
| **TBoostv3 (CatBoost)** | 0.8684 ± 0.0049 | 0.6725 ± 0.0129 | **+0.56%** | **+0.89%** |

### Key Findings

- **CatBoost achieves the best overall performance** with AUC of 0.8684 and AUC-PR of 0.6725
- Both XGBoost and CatBoost variants substantially outperform their baselines
- Enhanced model shows **improved stability** with lower standard deviation
- Global statistical features provide significant boost when combined with RNN embeddings

## Key Features

### Model Components

- **Time-Embedded RNN**: Custom RNN cell that explicitly models temporal dynamics and time gaps in medical time series
- **Global Statistical Features**: 5 statistics (mean, max, min, std, slope) computed per temporal feature
- **Improved Gated Head**: Advanced pre-training architecture with feature gating and dropout for RNN training
- **Focal Loss**: Addresses severe class imbalance with label smoothing to prevent overfitting
- **Triple Feature Fusion**: Synergistic combination of point-wise, statistical, and sequential features
- **Mixed Precision Training**: Automatic FP16/FP32 computation for faster training

### Training Enhancements

- **Focal Loss with Label Smoothing**: Better handling of class imbalance than weighted BCE
- **Advanced Regularization**: Feature dropout, weight decay, gradient clipping
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rate adjustment
- **Early Stopping**: Prevents overfitting with validation monitoring and patience mechanism
- **Overfitting Detection**: Automatic detection and early termination on severe overfitting

### Evaluation

- **Comprehensive Metrics**: AUC, AUC-PR, accuracy, specificity, precision, recall
- **Cross-Validation**: 5-fold stratified cross-validation for robust performance estimates
- **Visualization**: ROC curves for both enhanced and baseline models across all folds

## Requirements

```bash
numpy
pandas
matplotlib
torch
xgboost
catboost
scikit-learn
joblib
```

## Project Structure

```
TXGBoost/
├── run_with_new_data.py        # TBoostv3 with XGBoost backend
├── run_new_catboost.py         # TBoostv3 with CatBoost backend (Best Performance)
├── TBoostv3.py                 # Core model components and training logic
├── TimeEmbedding.py            # Time-embedded RNN cell
├── TimeEmbeddingVal.py         # Data preparation utilities
├── new_data_loader.py          # New data format loader with caching
├── new_data_helpers.py         # Feature extraction helpers
├── constants.py                # Feature definitions
├── utils/
│   ├── class_patient.py        # Patient data structure
│   └── prepare_data.py         # Data preprocessing
├── new_data/
│   └── final_dataset.pkl       # Dataset in new format
└── result/
    ├── tboostv2_new_data_vs_baseline.png          # XGBoost results
    └── tboostv2_new_data_vs_baseline_catboost.png # CatBoost results
```

## Usage

### Basic Usage

Run either gradient boosting variant:

```bash
# XGBoost variant
python run_with_new_data.py

# CatBoost variant (Recommended for best performance)
python run_new_catboost.py
```

### Model Selection Guide

- **CatBoost (run_new_catboost.py)**: Best performance (recommended for production)
  - Superior handling of categorical features
  - Better regularization for medical data
  - Highest AUC and AUC-PR scores

- **XGBoost (run_with_new_data.py)**: Faster training with strong performance
  - Faster iterations during development
  - Slightly lower but still excellent performance
  - Good for rapid experimentation

## Model Pipeline

### Stage 1: RNN Pre-training

The time-embedded RNN is pre-trained using an improved gated decision head:

```python
# Pre-train RNN with [RNN (256) + Enhanced Static] → Improved Gated Head
rnn = RNNFeatureExtractor(input_dim=temporal_feats, hidden_dim=256)
rnn = train_rnn_extractor_new_format(
    rnn, train_loader, val_loader,
    criterion=FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.15),
    static_dim=enhanced_static_dim,
    epochs=100
)
```

**Key improvements:**
- Focal Loss with label smoothing for severe class imbalance
- Improved Gated Head with feature dropout and batch normalization
- Mixed precision training (FP16/FP32) for faster computation
- Learning rate scheduling with ReduceLROnPlateau
- Advanced early stopping with overfitting detection

### Stage 2: Feature Extraction

Extract triple features combining last values, enhanced static features, and RNN embeddings:

```python
# [Last Values + Enhanced Static + RNN Embedding] = Total dims
# For example: [25 + 148 + 256] = 429 dims (varies by dataset)
X_train, y_train = get_triple_features(rnn, train_loader)
```

### Stage 3: Gradient Boosting Training

Train gradient boosting classifier on the fused features:

**XGBoost:**
```python
clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    eval_metric='auc',
    random_state=42
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

**CatBoost:**
```python
clf = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    eval_metric='AUC',
    random_seed=42
)
clf.fit(X_train, y_train, eval_set=(X_val, y_val))
```

## Static Features

The model automatically detects static features from the dataset. Typical static features include:

**Demographics**: age, gender, race

**Comorbidities**: chronic_pulmonary_disease, ckd_stage, congestive_heart_failure, dka_type, history_aci, history_ami, hypertension, liver_disease, macroangiopathy, malignant_cancer, microangiopathy, uti

**Severity Scores**: oasis, saps2, sofa

**Interventions**: mechanical_ventilation, use_NaHCO3, preiculos, gcs_unable

## Temporal Features

The model automatically detects and processes temporal features from medical time series data. For each temporal feature, the following are computed:

### Last Values
Most recent observation for each temporal variable (provides strong baseline signal)

### Global Statistical Features

Five statistical features computed over the entire observation window:

1. **Mean**: Average value over the observation window
2. **Max**: Peak value (important for detecting critical events)
3. **Min**: Lowest value (important for detecting concerning drops)
4. **Std**: Variability/stability of the measurement
5. **Slope**: Trend direction `(last - first) / (time_last - time_first)`

### RNN Embeddings

The Time-Embedded RNN learns complex temporal patterns including:
- Sequential dependencies between observations
- Temporal decay based on time gaps
- Irregular sampling patterns
- Missing data handling via masking

## Key Implementation Details

### Time-Embedded RNN Cell

Custom RNN that explicitly models time gaps between observations:

```python
class TimeEmbeddedRNNCell:
    # Learns temporal decay functions
    # Handles irregular time series
    # Accounts for missing data via masking
    # Outputs 256-dimensional temporal embeddings
```

### Improved Gated Decision Head

Advanced pre-training head with anti-overfitting measures:

```python
class ImprovedGatedHead:
    # Feature gating layer (learns which features matter)
    # Batch normalization for stable training
    # Feature dropout (0.4 × dropout rate)
    # Deep encoder with progressive dimension reduction
    # Outputs logits for Focal Loss
```

### Focal Loss with Label Smoothing

Addresses severe class imbalance:

```python
class FocalLoss:
    # alpha: Balances positive/negative examples (typically 0.25-0.75)
    # gamma: Focuses on hard examples (typically 2.0)
    # label_smoothing: Prevents overconfidence (typically 0.15)
    # Much better than weighted BCE for severe imbalance
```

### Data Loading with Caching

Efficient data loading with automatic caching:

```python
patients = load_and_prepare_new_format_patients(
    data_filepath,
    min_feature_coverage=0.8,  # Require 80% feature coverage
    use_cache=True  # Cache processed data for faster subsequent runs
)
```

## Configuration

### Key Hyperparameters

**RNN Training:**
- Hidden dimension: 256
- Batch size: 128
- Learning rate: 0.0005 (with ReduceLROnPlateau scheduler)
- Dropout: 0.5
- Epochs: 100 (with early stopping, patience=6)
- Loss: Focal Loss (alpha=0.25-0.75, gamma=2.0, label_smoothing=0.15)

**Gradient Boosting:**
- Iterations/n_estimators: 500
- Max depth: 6
- Learning rate: 0.05
- Scale pos weight: Automatic based on class ratio

**Data Loading:**
- Batch size: 128
- Num workers: 4
- Pin memory: True
- Persistent workers: True
- Mixed precision: Enabled

## Output

Each script generates:

1. Console output with detailed fold-by-fold results
2. ROC curves comparison plot in `result/` directory
3. Final performance statistics with mean ± std across folds
4. Feature dimension analysis and model configuration details

## Why This Approach Works

1. **Multi-Scale Information**: Combines point-wise (last), sequential (RNN), and aggregate (stats) views
2. **Complementary Features**: RNN learns temporal patterns that static features miss
3. **Explicit Statistical Features**: Capture aggregate trends and data quality signals
4. **Gradient Boosting Strengths**: Excels at combining heterogeneous features and handling non-linearities
5. **Pre-training Strategy**: Improved Gated Head ensures RNN learns boosting-compatible representations
6. **Advanced Regularization**: Focal Loss, label smoothing, and feature dropout prevent overfitting
7. **CatBoost Advantages**: Better handling of categorical features and stronger regularization for medical data

## Citation

If you use this code in your research, please cite:

```
[Add your citation here]
```

## License

[Add your license here]

## Contact

[Add your contact information here]
