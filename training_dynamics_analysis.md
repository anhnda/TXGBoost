# Training Dynamics Analysis: XGRL vs XGRLv2

## The Real Problem: Training Flow Mismatch

Since enhanced features (Last + Mean + Std) work fine for XGBoost in other work, the issue is **how the RL training interacts with the policy**.

## Critical Differences in Training Flow

### XGRL.py (Works)
```python
# 1. NO PRETRAINING - starts from random policy
# 2. Direct RL from epoch 0
# 3. Learning rate: 0.0005
# 4. Reward: 0.5 * binary + 0.5 * smooth_prob
# 5. XGBoost updates every 5 epochs
# 6. Entropy bonus: 0.01
# 7. Early stopping on: val_aupr
```

### XGRLv2.py (Fails)
```python
# 1. PRETRAINING: 50 epochs with SupervisedHead + BCE loss
# 2. Then RL starts
# 3. Learning rate: 0.0001 (5x smaller!)
# 4. Reward: smooth_prob + 0.5 * val_aupr (global metric added)
# 5. XGBoost updates every 5 epochs
# 6. Entropy bonus: 0.001 (10x smaller!)
# 7. Temperature annealing: 0.8 → 0.3
# 8. Early stopping on: val_aupr
```

## Root Causes of Failure

### 1. **Pretraining Creates Wrong Initialization**
```python
# Pretraining optimizes: minimize BCE(SupervisedHead(Z + Static), labels)
# RL optimizes: maximize XGBoost_reward(XGB([Static + Last + Mean + Std + Z]))
```

**Problem**:
- Pretraining learns Z for a **neural network** (SupervisedHead)
- RL needs Z for a **tree-based model** (XGBoost)
- These have completely different feature preferences!
- Neural nets: smooth, continuous representations
- Trees: sharp, decision-boundary-aligned features

**Result**: Policy gets stuck in a local optimum that's good for neural nets but bad for trees.

### 2. **Learning Rate Too Small After Pretraining**
```python
# XGRL: lr = 0.0005 (fresh start, can explore)
# XGRLv2: lr = 0.0001 (after pretraining, can't escape)
```

**Problem**:
- After 50 epochs of pretraining, policy is confidently wrong
- LR = 0.0001 is too small to escape the pretrained local optimum
- RL signal is weak compared to pretrained weights

### 3. **Global Reward Signal is Non-Stationary**
```python
# XGRLv2: rewards = smooth_prob + 0.5 * val_aupr
#                                    ^^^^^^^^^^^^^^ SAME for all samples!
```

**Problem**:
- val_aupr changes every epoch → reward function is non-stationary
- Policy gradient assumes stationary reward distribution
- This violates RL assumptions and destabilizes training

**Example**:
```
Epoch 1: val_aupr = 0.70 → all samples get +0.35 reward bonus
Epoch 2: val_aupr = 0.75 → all samples get +0.375 reward bonus
Epoch 3: val_aupr = 0.68 → all samples get +0.34 reward bonus
```
The same policy action gets different rewards just because val_aupr fluctuates!

### 4. **Temperature Annealing Reduces Exploration Too Early**
```python
# XGRLv2: temperature = max(0.3, 0.8 - epoch / (epochs * 0.7))
# At epoch 56 (of 80): temp = 0.3 (very low exploration)
```

**Problem**:
- Combined with small LR, policy can't explore enough
- Gets stuck in pretrained local optimum
- TabPFN works because it's already pre-trained on millions of datasets
- XGBoost has no such prior → needs MORE exploration, not less

### 5. **Entropy Bonus Too Weak**
```python
# XGRL: entropy_bonus = 0.01 * log_probs.mean()
# XGRLv2: entropy_bonus = 0.001 * log_probs.mean() (10x weaker!)
```

**Problem**:
- Weak entropy → policy becomes too deterministic
- Can't escape pretrained initialization
- Exploration dies out quickly

## Why TabPFNRL Works Despite These Issues

TabPFN is special because:
1. **Pre-trained on millions of datasets** → pretrained initialization is actually GOOD
2. **Small dataset (1024 limit)** → less data to overfit on
3. **Robust architecture** → can tolerate non-stationary rewards
4. **Few-shot learner** → designed for conservative exploration

XGBoost has NONE of these properties!

## Fixes for XGRLv2

### Option 1: Remove Pretraining (Easiest)
```python
# Just skip the pretrain_rnn_enhanced() step
# Start RL from random initialization
```

### Option 2: Fix the RL Training (Better)
```python
# 1. Remove global reward:
rewards_combined = rewards_smooth  # Remove "+ 0.5 * val_aupr"

# 2. Increase learning rate:
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0005)  # Not 0.0001

# 3. Increase entropy:
entropy_bonus = 0.01 * log_probs.mean()  # Not 0.001

# 4. Remove temperature annealing (or make it slower):
temperature = 1.0  # Fixed, or very slow decay

# 5. Use warm restart after pretraining:
# Reset optimizer after pretraining to forget momentum
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)  # Fresh optimizer
```

### Option 3: Fix Pretraining Objective (Best)
```python
# Pretrain with XGBoost as the head, not SupervisedHead!
def pretrain_with_xgboost(policy_net, train_loader, val_loader, xgb_params, epochs=30):
    """Pretrain to generate features that XGBoost likes"""
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    for epoch in range(epochs):
        # Extract deterministic features
        X_train, y_train, _ = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=True
        )

        # Train XGBoost
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)

        # Get feature importance from XGBoost
        importance = xgb.feature_importances_

        # Reward policy for generating important features
        # (This requires a differentiable surrogate - complex!)
        # ... OR use simpler supervised objective with XGBoost predictions as targets
```

## The Key Insight

**The failure is not about features, it's about the training dynamics:**

1. Pretraining optimizes for neural network → wrong manifold
2. Small LR + weak entropy + temperature annealing → can't escape
3. Non-stationary global rewards → unstable training
4. These issues don't affect TabPFN because it's already pre-trained externally

**Solution**: Either skip pretraining, or fix the RL training to allow escape from the pretrained initialization.
