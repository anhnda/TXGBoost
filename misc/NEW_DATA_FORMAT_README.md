# New Data Format Loader for TXGBoost

This document explains how to use the new data format with the TXGBoost pipeline.

## Overview

The new data loader (`new_data_loader.py`) converts joblib-dumped dictionary data into the `Patient`/`Patients` object structure expected by the existing TXGBoost pipeline.

## Data Format Comparison

### Old Format (Original)
```python
[
    {
        "subjectId": 19277038,
        "hadmId": 20027438,
        "stayId": 30213832,
        "akdPositive": false,
        "measures": {
            "hr": {
                "2119-04-08T00:04:00": 106.0,
                "2119-04-08T00:18:00": 99.0,
                ...
            },
            "age": 64,
            "gender": "F",
            ...
        }
    },
    ...
]
```

### New Format
```python
{
    "30000213": {
        "heart_rate": [
            {"charttime": "2162-06-21 05:46:00", "value": 74.0},
            {"charttime": "2162-06-21 06:00:00", "value": 74.0},
            ...
        ],
        "age_at_admission": 66,
        "gender": "M",
        "target": 0,
        ...
    },
    ...
}
```

## Key Differences

1. **Structure**:
   - Old: List of patient dicts
   - New: Dictionary with stay_id as key

2. **Temporal Data**:
   - Old: `{timestamp_string: value}`
   - New: `[{"charttime": ..., "value": ...}, ...]`

3. **Field Names**:
   - Old: `akdPositive`, `subjectId`, `hr`, `scr`
   - New: `target`, stay_id as key, `heart_rate`, `creatinine`

4. **Feature Set**:
   - New format has additional clinical features (sepsis, hepatitis, etc.)
   - Some features are renamed or reorganized

## Feature Name Mappings

The loader automatically maps feature names:

| New Format | Old Format | Type |
|------------|------------|------|
| `heart_rate` | `hr` | Temporal |
| `resp_rate` | `rr` | Temporal |
| `creatinine` | `scr` | Temporal |
| `glucose` | `bg` | Temporal |
| `age_at_admission` | `age` | Static |
| `target` | `akdPositive` | Label |
| `sapsii` | `saps2` | Static |
| ... | ... | ... |

See `FEATURE_NAME_MAPPING` in `new_data_loader.py` for the complete list.

## Usage

### Basic Usage

```python
from new_data_loader import load_and_prepare_new_format_patients

# Load and preprocess data
patients = load_and_prepare_new_format_patients('data/patients.joblib')

# Now use with existing TXGBoost pipeline
print(f"Loaded {len(patients)} patients")
```

### Running the Full Pipeline

```bash
# Using the integration script
python run_with_new_data.py data/patients_new_format.joblib
```

### Programmatic Usage

```python
from new_data_loader import load_and_prepare_new_format_patients
from TBoostv2 import main as run_tboostv2

# Load data
patients = load_and_prepare_new_format_patients(
    'data/patients.joblib',
    min_feature_coverage=0.8  # Remove features with <80% coverage
)

# Use with existing pipeline (requires minor modifications to TBoostv2.py)
# Replace the load_and_prepare_patients() call with the above
```

### Custom Preprocessing

```python
from new_data_loader import load_new_format_data
from constants import NULLABLE_MEASURES

# Load without preprocessing
patients = load_new_format_data('data/patients.joblib')

# Custom preprocessing
patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)
patients.removeMeasures(['feature_to_remove'])
patients.removePatientByMissingFeatures()

# Continue with pipeline...
```

## Integration with Existing Pipeline

### Option 1: Use Integration Script (Recommended)

```bash
python run_with_new_data.py path/to/data.joblib
```

This script:
- Loads data in new format
- Converts to Patient objects
- Runs the full TBoostv2 pipeline
- Generates comparison plots and metrics

### Option 2: Modify Existing Scripts

Replace the data loading section in `TBoostv2.py`:

```python
# OLD:
from TimeEmbedding import load_and_prepare_patients
patients = load_and_prepare_patients()

# NEW:
from new_data_loader import load_and_prepare_new_format_patients
patients = load_and_prepare_new_format_patients('data/patients.joblib')
```

## Data Loader Functions

### `load_new_format_data(filepath)`

Loads raw data and converts to Patient objects without preprocessing.

**Args:**
- `filepath`: Path to joblib file

**Returns:**
- `Patients` object

### `load_and_prepare_new_format_patients(filepath, nullable_measures, min_feature_coverage)`

Loads and preprocesses data (recommended).

**Args:**
- `filepath`: Path to joblib file
- `nullable_measures`: List of features to fill with 0 if missing (default: uses `NULLABLE_MEASURES`)
- `min_feature_coverage`: Minimum fraction of patients required to keep feature (default: 0.8)

**Returns:**
- Preprocessed `Patients` object

### `convert_patient_data(stay_id_str, patient_data)`

Converts a single patient record from new format to `Patient` object.

**Args:**
- `stay_id_str`: Stay ID as string
- `patient_data`: Patient data dictionary

**Returns:**
- `Patient` object

## Feature Coverage and Preprocessing

The loader automatically:

1. **Converts temporal data**: List of dicts → dict of timestamps
2. **Maps feature names**: New names → old names
3. **Extracts IDs**: Derives subject_id and hadm_id from stay_id
4. **Handles data types**: Converts booleans, strings, floats appropriately
5. **Filters features**: Removes features with low coverage (<80% by default)
6. **Removes patients**: Removes patients with missing critical features

## Handling Missing Features

### Features Not in Old Format

Some features in the new format don't exist in the old pipeline and are automatically skipped:

- `specimen`, `pao2fio2ratio`, `totalco2`, `baseexcess`, `ph`
- `d_dimer`, `inr`, `pt`, `ptt`
- `hepatitis`, `ventricular_arrhythmia`, `atrial_fibrillation`
- Many others (see `SKIP_FEATURES` in code)

### Features Not in New Format

If old format features are missing, they'll be handled by the pipeline's existing missing data logic:

- Nullable features (from `NULLABLE_MEASURES`) are filled with 0
- Features with low coverage are removed
- Patients with too many missing features are excluded

## Testing the Loader

```bash
# Test the loader independently
python new_data_loader.py data/patients.joblib
```

This will:
- Load the data
- Print conversion statistics
- Show sample patient information
- Report any conversion errors

## Expected Output Structure

After loading, each `Patient` object has:

```python
Patient(
    subject_id=30000,      # Derived from stay_id
    hadm_id=300,           # Derived from stay_id
    stay_id=30000213,      # From dictionary key
    intime=Timestamp(...), # From first temporal measurement
    akdPositive=False,     # From 'target' field
    measures={
        'hr': {Timestamp(...): 74.0, ...},  # Temporal
        'age': 66,                            # Static
        'gender': 'M',                        # Static
        ...
    }
)
```

## Troubleshooting

### Error: "Data file not found"
- Check the file path is correct
- Ensure the file is a joblib dump

### Error: "Expected dict, got ..."
- Verify the file contains a dictionary
- Check it's in the new format structure

### Warning: "N patients failed conversion"
- Some patient records may be incomplete
- Check the conversion logs for details
- The pipeline will continue with successfully converted patients

### Low AKI Positive Rate
- Check that 'target' field is correctly mapped
- Verify the data has balanced classes
- Review feature mappings for clinical relevance

## Performance Considerations

- **Memory**: Loading large datasets may require significant RAM
- **Conversion Time**: ~1000 patients/second on typical hardware
- **Feature Coverage**: Setting `min_feature_coverage` too high may remove important features

## Example: Complete Workflow

```python
# 1. Load data
from new_data_loader import load_and_prepare_new_format_patients

patients = load_and_prepare_new_format_patients(
    'data/aki_patients.joblib',
    min_feature_coverage=0.75
)

print(f"Loaded {len(patients)} patients")

# 2. Extract features
from TimeEmbeddingVal import get_all_temporal_features

temporal_features = get_all_temporal_features(patients)
print(f"Using {len(temporal_features)} temporal features")

# 3. Run model
from run_with_new_data import main

main('data/aki_patients.joblib')

# 4. Check results
# - Plot: result/tboostv2_new_data_vs_baseline.png
# - Metrics printed to console
```

## File Structure

```
TXGBoost/
├── new_data_loader.py              # New data format loader
├── run_with_new_data.py            # Integration script
├── TBoostv2.py                     # Original pipeline
├── NEW_DATA_FORMAT_README.md       # This file
├── example/
│   ├── Old_example_bio.txt         # Old format example
│   └── New_example_bio.txt         # New format example
└── result/
    └── tboostv2_new_data_vs_baseline.png  # Output plots
```

## Extending the Loader

To add new feature mappings:

```python
# In new_data_loader.py, add to FEATURE_NAME_MAPPING:
FEATURE_NAME_MAPPING = {
    # ... existing mappings ...
    'new_feature_name': 'old_feature_name',
}
```

To handle special feature types:

```python
# In convert_patient_data(), add custom logic:
elif old_feature_name == 'special_feature':
    measures[old_feature_name] = custom_transform(feature_value)
```

## Contact & Support

For issues or questions:
1. Check this README
2. Review example files in `example/`
3. Test with `python new_data_loader.py <file>`
4. Check conversion warnings in output

## Version Compatibility

- Compatible with TXGBoost v2
- Requires Python 3.7+
- Dependencies: pandas, numpy, joblib, torch, xgboost, scikit-learn
