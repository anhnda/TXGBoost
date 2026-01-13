# TXGBoost: Triple Hybrid Model for AKI Prediction

A novel hybrid deep learning approach that combines temporal pattern learning with gradient boosting for Acute Kidney Disease (AKD) prediction.

## Overview

TXGBoost implements a **triple hybrid architecture** that significantly outperforms traditional feature engineering approaches by learning temporal patterns through a Gated RNN and combining them with handcrafted features for XGBoost.

### Architecture

```
[Learned Temporal Trends + Last Values + Static Context] → XGBoost
```

The model combines three complementary feature types:
1. **Learned Trends (128 dims)**: Time-embedded RNN captures temporal dynamics from medical time series
2. **Last Values (25 dims)**: Most recent measurements of temporal features
3. **Static Context (23 dims)**: Patient demographics and clinical history

This is compared against a **strong baseline**:
```
[Last Values + Static Context] → XGBoost
```

## Performance

The triple hybrid approach demonstrates significant improvements over the baseline:

| Metric          | Hybrid Model          | Baseline Model        | Improvement |
|-----------------|----------------------|----------------------|-------------|
| **AUC**         | 0.8426 ± 0.0204      | 0.8192 ± 0.0237      | +2.34%      |
| **AUC-PR**      | 0.7820 ± 0.0268      | 0.7449 ± 0.0515      | +3.71%      |

Results are averaged over 5-fold cross-validation.

## Key Features

- **Time-Embedded RNN**: Custom RNN cell that explicitly models temporal dynamics in medical time series
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
├── TXGBoost.py              # Main implementation
├── TimeEmbedding.py         # Time-embedded RNN cell
├── TimeEmbeddingVal.py      # Data preparation utilities
├── constants.py             # Feature definitions
├── utils/
│   ├── class_patient.py     # Patient data structure
│   └── prepare_data.py      # Data preprocessing
└── result/
    └── triple_hybrid_vs_baseline.png  # ROC curves
```

## Usage

### Basic Usage

```bash
python TXGBoost.py
```

### Custom Configuration

```python
from TXGBoost import main

# The script uses the following key parameters:
# - RNN hidden dimension: 128
# - XGBoost: n_estimators=500, max_depth=6, learning_rate=0.05
# - Batch size: 32
# - RNN pre-training epochs: 50 (with early stopping)
# - Random seed: 42
```

## Model Pipeline

### Stage 1: RNN Pre-training
The time-embedded RNN is pre-trained using a gated decision head that mimics XGBoost's decision-making process:

```python
# Pre-train RNN with [RNN + Static] → Gated Head
model = RNNFeatureExtractor(input_dim=25, hidden_dim=128)
model = train_rnn_extractor(model, train_loader, val_loader, epochs=50)
```

### Stage 2: Triple Feature Extraction
Extract three complementary feature sets:

```python
# [Last Values (25) + Static (23) + RNN Embedding (128)] = 176 total dims
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

## Why This Approach Works

1. **Complementary Features**: RNN learns temporal patterns that static features miss
2. **Explicit Last Values**: Provides strong baseline signal to XGBoost
3. **XGBoost Strengths**: Excels at combining heterogeneous features and handling non-linearities
4. **Pre-training Strategy**: Gated head ensures RNN learns XGBoost-compatible representations

## Citation

If you use this code in your research, please cite:

```
[Add your citation here]
```

## License

[Add your license here]

## Contact

[Add your contact information here]
