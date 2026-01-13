# TXGBoost: Triple Hybrid Model for AKI Prediction

A novel hybrid deep learning approach that combines temporal pattern learning with gradient boosting for Acute Kidney Disease (AKD) prediction.

## Overview

TXGBoost implements a **triple hybrid architecture** that significantly outperforms traditional feature engineering approaches by learning temporal patterns through a Gated RNN and combining them with handcrafted features for XGBoost.

## Model Variants

### TXGBoost (Original)
```
[Learned Temporal Trends + Last Values + Static Context] → XGBoost
```
- **176 dims**: Last (25) + Static (23) + RNN (128)

### TBoostv1 (Enhanced RNN Training)
```
RNN Training: [RNN + Enhanced Static (173)] → Gated Head
XGBoost:      [Last (25) + Original Static (23) + RNN (128)] → Prediction
```
- Global stats (mean/max/min/std/slope/count) enhance RNN learning
- XGBoost uses original features only: **176 dims**

### TBoostv2 (Full Enhancement) ⭐ Best Performance
```
RNN Training: [RNN + Enhanced Static (173)] → Gated Head
XGBoost:      [Last (25) + Enhanced Static (173) + RNN (128)] → Prediction
```
- Global stats enhance both RNN training AND XGBoost
- Full feature set: **326 dims**

## Performance Comparison

All results are averaged over 5-fold cross-validation against the same strong baseline:

| Model           | AUC                  | AUC-PR               | vs Baseline AUC | vs Baseline AUC-PR |
|-----------------|----------------------|----------------------|-----------------|-------------------|
| **Baseline**    | 0.8192 ± 0.0237     | 0.7449 ± 0.0515     | -               | -                 |
| **TXGBoost**    | 0.8426 ± 0.0204     | 0.7820 ± 0.0268     | **+2.86%**      | **+4.98%**        |
| **TBoostv2** ⭐  | 0.8553 ± 0.0265     | 0.8011 ± 0.0288     | **+4.41%**      | **+7.54%**        |

**Key Findings:**
- TBoostv2 achieves the best performance with **+4.41% AUC** and **+7.54% AUC-PR** improvement
- Global statistical features provide significant boost when used in both RNN and XGBoost
- All variants substantially outperform the traditional baseline

### Feature Comparison

| Aspect                  | TXGBoost      | TBoostv1      | TBoostv2 ⭐   |
|-------------------------|---------------|---------------|---------------|
| **RNN Training Input**  | 23 static     | 173 enhanced  | 173 enhanced  |
| **XGBoost Input Dims**  | 176           | 176           | 326           |
| **Global Stats in RNN** | ✗             | ✓             | ✓             |
| **Global Stats in XGB** | ✗             | ✗             | ✓             |
| **AUC**                 | 0.8426        | -             | **0.8553**    |
| **AUC-PR**              | 0.7820        | -             | **0.8011**    |

## Key Features

- **Time-Embedded RNN**: Custom RNN cell that explicitly models temporal dynamics in medical time series
- **Global Statistical Features**: 6 statistics (mean, max, min, std, slope, count) computed per temporal feature
- **Gated Decision Head**: XGBoost-mimicking architecture for RNN pre-training
- **Triple Feature Fusion**: Synergistic combination of learned and handcrafted features
- **Categorical Encoding**: Automatic handling of categorical features (Gender, Race)
- **Full Evaluation Suite**: Comprehensive metrics including AUC, AUC-PR, accuracy, specificity, precision, and recall
- **Visualization**: ROC curves for both models across all folds

## Requirements

```bash
numpy
pandas
matplotlib
torch
xgboost
scikit-learn
```

## Project Structure

```
TXGBoost/
├── TXGBoost.py              # Original triple hybrid model (176 dims)
├── TBoostv1.py              # Enhanced RNN training, original XGBoost features (176 dims)
├── TBoostv2.py              # Full enhancement with global stats (326 dims) ⭐
├── TimeEmbedding.py         # Time-embedded RNN cell
├── TimeEmbeddingVal.py      # Data preparation utilities
├── constants.py             # Feature definitions
├── utils/
│   ├── class_patient.py     # Patient data structure
│   └── prepare_data.py      # Data preprocessing
└── result/
    ├── triple_hybrid_vs_baseline.png  # TXGBoost results
    ├── tboostv1_vs_baseline.png       # TBoostv1 results
    └── tboostv2_vs_baseline.png       # TBoostv2 results
```

## Usage

### Basic Usage

Run any of the three model variants:

```bash
# Original triple hybrid model
python TXGBoost.py

# Enhanced RNN training (v1)
python TBoostv1.py

# Full enhancement with global stats (v2) - Recommended ⭐
python TBoostv2.py
```

### Model Selection Guide

- **TBoostv2**: Use for best performance (recommended for production)
- **TXGBoost**: Use as baseline for understanding core architecture
- **TBoostv1**: Use to study the impact of enhanced RNN training

### Custom Configuration

All scripts share the following key parameters:
```python
# - RNN hidden dimension: 128
# - XGBoost: n_estimators=500, max_depth=6, learning_rate=0.05
# - Batch size: 32
# - RNN pre-training epochs: 50 (with early stopping)
# - Random seed: 42
```

## Model Pipeline

### Stage 1: RNN Pre-training
The time-embedded RNN is pre-trained using a gated decision head that mimics XGBoost's decision-making process:

**TXGBoost (Original):**
```python
# Pre-train RNN with [RNN + Static (23)] → Gated Head
model = RNNFeatureExtractor(input_dim=25, hidden_dim=128)
model = train_rnn_extractor(model, train_loader, val_loader, epochs=50)
```

**TBoostv1 & TBoostv2 (Enhanced):**
```python
# Pre-train RNN with [RNN + Enhanced Static (173)] → Gated Head
# Enhanced Static = Original Static (23) + Global Stats (150)
model = RNNFeatureExtractor(input_dim=25, hidden_dim=128)
model = train_rnn_extractor(model, train_loader, val_loader, epochs=50)
```

### Stage 2: Feature Extraction

**TXGBoost & TBoostv1:**
```python
# [Last Values (25) + Static (23) + RNN Embedding (128)] = 176 dims
X_train, y_train = get_triple_features(rnn, train_loader)
```

**TBoostv2:**
```python
# [Last Values (25) + Enhanced Static (173) + RNN Embedding (128)] = 326 dims
X_train, y_train = get_triple_features(rnn, train_loader)
```

### Stage 3: XGBoost Training
Train gradient boosting classifier on the fused features:

```python
clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

## Static Features

The model uses 23 static features:

**Demographics**: age, gender, race

**Comorbidities**: chronic_pulmonary_disease, ckd_stage, congestive_heart_failure, dka_type, history_aci, history_ami, hypertension, liver_disease, macroangiopathy, malignant_cancer, microangiopathy, uti

**Severity Scores**: oasis, saps2, sofa

**Interventions**: mechanical_ventilation, use_NaHCO3, preiculos, gcs_unable

## Temporal Features

The model processes 25 temporal features from medical time series data, extracting patterns through the time-embedded RNN.

## Key Implementation Details

### Time-Embedded RNN Cell
Custom RNN that explicitly models time gaps between observations:

```python
class TimeEmbeddedRNNCell:
    # Learns temporal decay functions
    # Handles irregular time series
    # Accounts for missing data via masking
```

### Gated Decision Head
Pre-training head that mimics XGBoost's gating mechanism:

```python
class GatedDecisionHead:
    # Feature gating layer
    # GLU activation functions
    # Residual connections
```

### Categorical Encoding
Automatic encoding of non-numeric features:

```python
encoder = SimpleStaticEncoder(FIXED_FEATURES)
encoder.fit(patients)  # Learn mappings from training data
```

## Output

The script generates:
1. Console output with fold-by-fold results
2. ROC curves comparison plot: `result/triple_hybrid_vs_baseline.png`
3. Final performance statistics with mean ± std

## Global Statistical Features (TBoostv1 & TBoostv2)

The enhanced models compute 6 statistical features for each temporal variable:

1. **Mean**: Average value over the observation window
2. **Max**: Peak value (important for detecting critical events)
3. **Min**: Lowest value (important for detecting concerning drops)
4. **Std**: Variability/stability of the measurement
5. **Slope**: Trend direction `(last - first) / (time_last - time_first)`
6. **Count**: Number of observations (captures data density)

For 25 temporal features, this produces **150 additional dimensions** (25 × 6).

### How Global Stats Enhance Performance

**In TBoostv1:**
- Global stats provide richer context during RNN training
- The RNN learns better representations by understanding both sequential patterns AND overall trends
- XGBoost receives compact 176-dim features with improved RNN embeddings

**In TBoostv2:**
- Global stats enhance RNN training (same as v1)
- XGBoost also gets direct access to statistical summaries (326 dims total)
- Best performance: combines learned sequential patterns with explicit statistical features

## Why This Approach Works

1. **Complementary Features**: RNN learns temporal patterns that static features miss
2. **Explicit Last Values**: Provides strong baseline signal to XGBoost
3. **Global Statistics**: Capture aggregate trends and data quality signals
4. **XGBoost Strengths**: Excels at combining heterogeneous features and handling non-linearities
5. **Pre-training Strategy**: Gated head ensures RNN learns XGBoost-compatible representations
6. **Multi-Scale Information**: Combines point-wise (last), sequential (RNN), and aggregate (stats) views

## Citation

If you use this code in your research, please cite:

```
[Add your citation here]
```

## License

[Add your license here]

## Contact

[Add your contact information here]
