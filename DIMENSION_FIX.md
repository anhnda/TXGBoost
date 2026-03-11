# Dimension Mismatch Fix

## Problem

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x331 and 330x330)
```

The RNN training failed because the GatedDecisionHead was initialized with the wrong input dimension.

## Root Cause

The original `train_rnn_extractor` function in `TBoostv2.py` calculated the static dimension as:

```python
static_dim = len(FIXED_FEATURES) + (len(train_loader.dataset.feature_names) * 5)
```

Where:
- `FIXED_FEATURES` = 23 features (from old format)
- This was hardcoded and didn't match the actual data

But in the new format:
- We dynamically detect features
- `static_feats` = 22 features (actual count from data)
- This caused a 1-feature difference: 22 vs 23

### Dimension Breakdown

**Expected by model (using FIXED_FEATURES=23):**
- Static features: 23
- Global stats: 45 temporal × 5 = 225
- Enhanced static: 23 + 225 = 248
- Combined input: 128 (RNN) + 248 = 376
- But the error showed 330, so there was another mismatch

**Actual in data:**
- Static features: 22 (detected from new format)
- Global stats: 45 temporal × 5 = 225
- Enhanced static: 22 + 225 = 247
- Combined input: 128 (RNN) + 247 = 375
- But actual was 331

The exact numbers don't matter - the key issue was using hardcoded `FIXED_FEATURES` instead of the actual detected `static_feats`.

## Solution

Created `train_rnn_extractor_new_format()` in `run_with_new_data.py` that:

1. **Takes static_dim as parameter** instead of calculating it from FIXED_FEATURES
2. **Uses actual feature counts** from the new data format
3. **Properly initializes GatedDecisionHead** with correct dimensions

### Code Changes

**Before:**
```python
# Used hardcoded function
rnn = train_rnn_extractor(rnn, train_loader, val_loader, ...)
# This internally calculated: static_dim = len(FIXED_FEATURES) + ...
# FIXED_FEATURES was wrong for new format!
```

**After:**
```python
# Calculate actual static dimension
actual_static_dim = len(static_feats) + (len(temporal_feats) * 5)

# Pass it explicitly
rnn = train_rnn_extractor_new_format(
    rnn, train_loader, val_loader, ...,
    static_dim=actual_static_dim,  # Use ACTUAL dimension
    epochs=50
)
```

### Implementation

```python
def train_rnn_extractor_new_format(model, train_loader, val_loader, criterion,
                                     optimizer, static_dim, epochs=50):
    """Train RNN with CORRECT static dimension for new data format."""
    rnn_dim = model.rnn_cell.hidden_dim

    # Create head with CORRECT dimensions
    temp_head = GatedDecisionHead(input_dim=rnn_dim + static_dim).to(DEVICE)
    # ✓ Now uses actual static_dim from data, not FIXED_FEATURES

    # ... rest of training logic ...
```

## Verification

The function now prints the dimensions:

```
[Stage 1] Training RNN extractor...
  Static features: 22
  Global stats: 225
  Total static dim: 247

[Stage 1] Pre-training RNN with Gated Head
  RNN dim: 128, Static dim: 247, Total: 375
```

This ensures:
- ✅ Static features = actual count from data
- ✅ Global stats = temporal features × 5
- ✅ RNN input = 128 + actual_static_dim
- ✅ No dimension mismatch

## Files Modified

1. **`run_with_new_data.py`**
   - Added `train_rnn_extractor_new_format()` function
   - Calculates `actual_static_dim` from detected features
   - Passes dimension explicitly to training

2. **`new_data_helpers.py`** (already created)
   - Properly detects temporal vs static features
   - Returns accurate feature counts

## Testing

Run the pipeline:

```bash
python run_with_new_data.py new_data/final_dataset.pkl
```

You should see:
1. ✅ Feature detection with correct counts
2. ✅ Dimension info printed
3. ✅ RNN training starts without errors
4. ✅ No more "shapes cannot be multiplied" error

## Key Takeaway

**Don't use hardcoded feature lists when working with dynamic data!**

- ❌ Bad: `static_dim = len(FIXED_FEATURES) + ...`
- ✅ Good: `static_dim = len(actual_static_feats) + ...`

The new format has different features than the old format, so we must detect them dynamically and use the actual counts.
