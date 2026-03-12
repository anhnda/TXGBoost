# 🚀 Quick Start Guide - New Data Loader

## In 3 Steps

### Step 1: Test the Loader
```bash
python new_data_loader.py your_data.joblib
```

### Step 2: Run the Pipeline
```bash
python run_with_new_data.py your_data.joblib
```

### Step 3: Check Results
```bash
ls result/
# Look for: tboostv2_new_data_vs_baseline.png
```

---

## What You Get

✅ **Automatic Data Conversion**
- New format (joblib dict) → Patient objects
- All features preserved with their new names

✅ **Complete Pipeline**
- RNN feature extraction
- Triple feature fusion
- XGBoost classification
- Baseline comparison

✅ **Detailed Results**
- ROC curves
- AUC, AUC-PR, Accuracy, etc.
- Fold-by-fold metrics

---

## Important Notes

### Feature Names
The loader **uses new format names directly**:
- ✅ `heart_rate` (not `hr`)
- ✅ `creatinine` (not `scr`)
- ✅ `age_at_admission` (not `age`)

### Data Format Required
Your data should be:
- **File type**: Joblib dump
- **Structure**: Dictionary with stay_id as keys
- **Temporal**: List of `{charttime, value}` dicts
- **Static**: Scalar values

### Example Input
```python
{
    "30000213": {
        "heart_rate": [{"charttime": "2162-06-21 05:46:00", "value": 74.0}, ...],
        "age_at_admission": 66,
        "target": 0,
        ...
    },
    ...
}
```

---

## Customization

### Exclude Features
Edit `new_data_loader.py`:
```python
ADDITIONAL_SKIP_FEATURES = {
    'feature_to_exclude',
}
```

### Adjust Coverage Threshold
```python
from new_data_loader import load_and_prepare_new_format_patients

patients = load_and_prepare_new_format_patients(
    'data.joblib',
    min_feature_coverage=0.75  # Default is 0.8
)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `new_data_loader.py` | Core loader |
| `run_with_new_data.py` | Full pipeline |
| `test_new_loader.py` | Test suite |
| `NEW_DATA_FORMAT_README.md` | Full documentation |
| `IMPLEMENTATION_COMPLETE.md` | Detailed summary |
| `UPDATE_NOTE.md` | Migration guide |

---

## Need Help?

1. **Test conversion**: `python test_new_loader.py`
2. **Check your data**: `python new_data_loader.py your_data.joblib`
3. **Read docs**: See `NEW_DATA_FORMAT_README.md`

---

**Ready to go!** Just run: `python run_with_new_data.py your_data.joblib`
