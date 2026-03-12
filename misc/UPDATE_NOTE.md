# Update Note: New Data Loader

## Summary of Changes

The `new_data_loader.py` has been **updated to use features from the new format directly**, without mapping to old format names. This is a more flexible approach that leverages all available features in the new dataset.

## Key Changes

### Before (Feature Mapping Approach)
- Mapped new feature names to old format names
  - `heart_rate` → `hr`
  - `creatinine` → `scr`  - `age_at_admission` → `age`
- Many new features were skipped to match old format
- Limited to features that existed in old format

### After (Direct Feature Usage) ✅
- **Uses new feature names directly** without mapping
  - `heart_rate` stays as `heart_rate`
  - `creatinine` stays as `creatinine`
  - `age_at_admission` stays as `age_at_admission`
- **All features from new format are preserved** (except metadata)
- **More flexible and extensible**

## What This Means

### ✅ Advantages

1. **More Features Available**: All clinical features from the new dataset are used
2. **No Information Loss**: No features excluded due to mapping issues
3. **Simpler Code**: No complex feature name mapping logic
4. **Future-Proof**: Easy to add new features without updating mapping tables
5. **Better Performance**: More features = potentially better model performance

### ⚠️ Important Notes

1. **Feature Names Changed**: The pipeline will work with new feature names
   - Old code expecting `hr` needs to work with `heart_rate` instead
   - Old code expecting `scr` needs to work with `creatinine` instead

2. **Preprocessing Still Works**: The pipeline's preprocessing (coverage filtering, nullable handling) works automatically with whatever features are present

3. **FIXED_FEATURES List**: You may need to update `FIXED_FEATURES` in `TBoostv2.py` to match the new feature names, or dynamically detect static features

## Usage

### Loading Data

```python
from new_data_loader import load_and_prepare_new_format_patients

# Load data - uses NEW feature names directly
patients = load_and_prepare_new_format_patients('data.joblib')

# Check what features are available
measures = patients.getMeasures()
print(f"Available features: {list(measures.keys())}")
```

### Integration with Pipeline

The loader works seamlessly with the existing pipeline. The key change is that **feature names will be from the new format**, not the old format.

#### Option 1: Let Pipeline Auto-Detect Features

```python
# In TBoostv2.py or run_with_new_data.py:

# Get ALL temporal features dynamically
temporal_feats = get_all_temporal_features(patients)

# Get static features dynamically (instead of using FIXED_FEATURES)
sample_patient = patients.patientList[0]
static_features = [k for k, v in sample_patient.measures.items()
                   if not isinstance(v, dict)]
```

#### Option 2: Update FIXED_FEATURES List

```python
# Update FIXED_FEATURES to match new format names
FIXED_FEATURES = [
    "age_at_admission",  # was 'age'
    "gender",
    "race",
    "bmi",
    "sepsis",
    "hepatitis",
    "hypertension",
    "diabetes",
    "ckd",
    "stroke",
    "copd",
    "pneumonia",
    "sapsii",  # was 'saps2'
    "lods",    # was 'sofa'
    # Add more as needed
]
```

## Files Affected

### Updated
- ✅ `new_data_loader.py` - Now uses new feature names directly
- ✅ `test_new_loader.py` - Works with new approach
- ✅ `run_with_new_data.py` - Compatible (uses dynamic feature detection)

### May Need Updates
- ⚠️ `TBoostv2.py` - If you want to use specific static features, update `FIXED_FEATURES` list
- ⚠️ Any custom code that hardcodes old feature names

## Migration Guide

If you have existing code using the old feature names:

### 1. Update Feature References

```python
# OLD:
if 'hr' in patient.measures:
    heart_rate = patient.measures['hr']

# NEW:
if 'heart_rate' in patient.measures:
    heart_rate = patient.measures['heart_rate']
```

### 2. Dynamic Feature Detection (Recommended)

```python
# Instead of hardcoding feature names, detect them:
temporal_features = get_all_temporal_features(patients)
static_features = [k for k, v in patients.patientList[0].measures.items()
                   if not isinstance(v, dict)]
```

### 3. Exclude Specific Features (Optional)

If you want to exclude certain features:

```python
# In new_data_loader.py, add to ADDITIONAL_SKIP_FEATURES:
ADDITIONAL_SKIP_FEATURES = {
    'feature_to_exclude',
    'another_feature_to_exclude',
}
```

## Testing

Test the updated loader:

```bash
# Test with your data
python new_data_loader.py your_data.joblib

# Run test suite
python test_new_loader.py

# Run full pipeline
python run_with_new_data.py your_data.joblib
```

## Example Output

When loading data, you'll now see:

```
Loading data from data.joblib...
Loaded 1234 patient records
Successfully converted 1234 patients
Total unique features: 67
  Temporal features: 45
  Static features: 22
  AKI positive: 456 (36.95%)
  AKI negative: 778 (63.05%)

Preprocessing data...
  Filled 12 nullable measures with 0
  Removed 5 low-coverage features
  Removed 12 patients with missing features

Final dataset: 1222 patients
Final features: 62

Available features (first 20):
  age_at_admission
  albumin
  anemia
  aniongap
  atrial_fibrillation
  bicarbonate
  bmi
  bun
  calcium
  chloride
  ckd
  copd
  creatinine
  diabetes
  gender
  glucose
  heart_rate
  hematocrit
  hemoglobin
  hepatitis
  ... and 42 more
```

## Questions?

The new approach is:
- **Simpler**: No complex mapping tables
- **More flexible**: Works with any feature set
- **More powerful**: Uses all available features
- **Future-proof**: Easy to extend

If you need specific features from the old format, you can always add manual renaming in your preprocessing code, but the default behavior is to use new features directly.
