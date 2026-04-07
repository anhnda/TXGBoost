# Analysis: Why XGRL.py Works but XGRLv2.py Fails

## Key Differences

### XGRL.py (Good Performance ✓)
```
Features: [Static(23) + Last(25) + Z(16)] = 64 dims
Network: hidden_dim=12, latent_dim=16
Training: Direct RL (no pretraining)
Rewards: 0.5 * binary_correct + 0.5 * probability_smooth
Exploration: Fixed (no temperature annealing)
XGBoost: n_estimators=200, max_depth=4, lr=0.05
```

### XGRLv2.py (Bad Performance ✗) - Based on TabPFNRL.py
```
Features: [Static(23) + Last(25) + Mean(25) + Std(25) + Z(28)] = 126 dims
Network: hidden_dim=20, latent_dim=28
Training: 50 epochs supervised pretraining + RL
Rewards: probability_smooth + 0.5 * val_aupr (AUPR-focused)
Exploration: Temperature annealing (0.8 → 0.3)
XGBoost: n_estimators=500, max_depth=6, lr=0.05
```

## Why TabPFNRL Approach Fails for XGBoost

### 1. **Feature Over-Engineering**
- **TabPFN**: Pre-trained transformer, benefits from rich features (Last + Mean + Std)
- **XGBoost**: Tree-based, sees Last/Mean/Std as redundant/correlated
  - Trees split on single features
  - Mean/Std don't add much value when Last is available
  - **Result**: 126 dims cause overfitting vs 64 dims

### 2. **Supervised Pretraining is Harmful**
- **TabPFN**: Pre-trained on millions of datasets, benefits from alignment
- **XGBoost**: Needs to learn what features XGBoost actually wants
  - Pretraining steers policy toward supervised objectives
  - But XGBoost reward landscape is different from BCE loss
  - **Result**: Policy gets stuck in local optima from pretraining

### 3. **AUPR-Focused Rewards Too Noisy**
```python
# XGRLv2: rewards = smooth + 0.5 * val_aupr
# This adds GLOBAL metric to PER-SAMPLE rewards
```
- **TabPFN**: Robust to noise, pre-trained representations
- **XGBoost**: Sensitive to reward structure
  - val_aupr is a global metric, same for all samples
  - Doesn't provide per-sample gradient information
  - **Result**: Noisy reward signal confuses policy

### 4. **Temperature Annealing Limits Exploration**
```python
# XGRLv2: temperature = max(0.3, 0.8 - epoch / (epochs * 0.7))
```
- **TabPFN**: Small dataset (1024 limit), conservative exploration OK
- **XGBoost**: Larger dataset, needs more exploration
  - Conservative annealing (0.8→0.3) too restrictive
  - Policy can't explore enough to find XGBoost-friendly features
  - **Result**: Gets stuck in suboptimal policies

### 5. **Model Capacity Mismatch**
- **TabPFN**: Deep transformer, can leverage latent_dim=28
- **XGBoost**: Trees use features independently
  - latent_dim=28 vs 16: more parameters to overfit
  - Larger hidden_dim=20 vs 12: more overfitting risk
  - **Result**: Policy overfits to training folds

## Why TabPFNRL Works for TabPFN

1. **Pre-trained on millions of datasets** → knows how to use rich features
2. **Transformer architecture** → captures complex feature interactions
3. **Designed for small data** → 1024 sample limit matches conservative RL
4. **Robust to noise** → can handle AUPR-focused rewards
5. **Needs rich context** → benefits from Last + Mean + Std + Z

## Why XGRL.py Works for XGBoost

1. **Simple features** → [Static + Last + Z] matches XGBoost's tree-based nature
2. **No pretraining** → policy learns directly from XGBoost rewards
3. **Clean per-sample rewards** → binary_correct + probability gives clear signal
4. **Direct exploration** → no temperature tricks, just sample from policy
5. **Right-sized capacity** → latent_dim=16, hidden_dim=12 prevents overfitting

## The Fundamental Insight

**TabPFN and XGBoost are different beasts:**

| Aspect | TabPFN | XGBoost |
|--------|--------|---------|
| Architecture | Pre-trained Transformer | Gradient Boosted Trees |
| Feature Usage | Complex interactions | Independent splits |
| Data Efficiency | Excellent (few-shot) | Needs more data |
| Feature Preference | Rich, diverse | Simple, uncorrelated |
| Capacity | Very high | Moderate |

**The approach that works for one doesn't transfer to the other!**

## Recommendations

To improve XGBoost RL:
1. ✓ Keep features simple: [Static + Last + Z]
2. ✓ Skip supervised pretraining
3. ✓ Use per-sample rewards only
4. ✓ Allow natural exploration (no temperature tricks)
5. ✓ Keep model capacity reasonable

To adapt for TabPFN:
1. ✓ Add enriched features: [Static + Last + Mean + Std + Z]
2. ✓ Use supervised pretraining
3. ✓ AUPR-focused rewards work well
4. ✓ Conservative exploration is fine
5. ✓ Larger capacity helps
