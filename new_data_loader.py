"""
New Data Loader for TXGBoost Pipeline

Loads joblib-dumped dictionary data in the new format and converts it
to the Patient/Patients object structure expected by the existing pipeline.

IMPORTANT: This loader uses features from the NEW format directly, without
mapping to old feature names. This allows us to leverage all available features
in the new dataset.

New Format Structure:
    {
        'stay_id_string': {
            'temporal_feature': [{'charttime': '...', 'value': ...}, ...],
            'static_feature': value,
            'target': 0 or 1,
            ...
        },
        ...
    }

Patient Object Created:
    Patient(
        subject_id, hadm_id, stay_id,
        intime, akdPositive,
        measures={
            'temporal_feature': {Timestamp: value, ...},  # Uses NEW feature names
            'static_feature': value                        # Uses NEW feature names
        }
    )
"""

import joblib
import pandas as pd
from pandas import Timestamp, Timedelta
from pathlib import Path
from typing import Dict, List, Union, Any

from utils.class_patient import Patient, Patients


# Metadata fields to exclude (not clinical features)
EXCLUDE_FEATURES = {
    'target',          # This is the label, not a feature
    'duration',        # Metadata
    'specimen',        # Not a measurement value
}

# Optional: Features to explicitly skip if you want to exclude them
# Leave empty {} to include all features from new format
ADDITIONAL_SKIP_FEATURES = {
    # Add features here if you want to exclude them
    # Example: 'height', 'weight_admit',
}


def convert_temporal_list_to_dict(temporal_list: List[Dict[str, Any]]) -> Dict[Timestamp, float]:
    """
    Convert temporal data from new format to Patient object format.

    New: [{'charttime': '2162-06-21 08:27:00', 'value': 100.0}, ...]
    Patient format: {Timestamp('2162-06-21 08:27:00'): 100.0, ...}

    Args:
        temporal_list: List of dicts with 'charttime' and 'value' keys

    Returns:
        Dict mapping Timestamp to value
    """
    if not temporal_list:
        return {}

    result = {}
    for item in temporal_list:
        if 'charttime' in item and 'value' in item:
            try:
                timestamp = pd.to_datetime(item['charttime'])
                value = float(item['value'])
                result[timestamp] = value
            except (ValueError, TypeError):
                # Skip invalid entries
                continue

    return result


def extract_ids_from_stay_id(stay_id_str: str) -> tuple:
    """
    Extract subject_id, hadm_id, stay_id from stay_id string.

    In the new format, we only have stay_id. We'll generate placeholder values
    for subject_id and hadm_id since they're not critical for the model.

    Args:
        stay_id_str: Stay ID as string

    Returns:
        (subject_id, hadm_id, stay_id) tuple
    """
    stay_id = int(stay_id_str)
    # Use stay_id as basis for other IDs (placeholders)
    subject_id = stay_id // 1000  # Placeholder
    hadm_id = stay_id // 100      # Placeholder

    return subject_id, hadm_id, stay_id


def convert_patient_data(stay_id_str: str, patient_data: Dict[str, Any]) -> Patient:
    """
    Convert a patient record from new format to Patient object.

    Uses feature names from the NEW format directly - no name mapping.
    This allows the pipeline to work with all available features in the new dataset.

    Args:
        stay_id_str: Stay ID as string (dictionary key)
        patient_data: Patient data dictionary in new format

    Returns:
        Patient object compatible with existing pipeline
    """
    # Extract IDs
    subject_id, hadm_id, stay_id = extract_ids_from_stay_id(stay_id_str)

    # Extract target (akdPositive)
    akd_positive = bool(patient_data.get('target', 0))

    # Use first temporal measurement as intime (or use a default)
    intime = None
    for feature_value in patient_data.values():
        if isinstance(feature_value, list) and len(feature_value) > 0:
            if isinstance(feature_value[0], dict) and 'charttime' in feature_value[0]:
                intime = pd.to_datetime(feature_value[0]['charttime'])
                break

    if intime is None:
        # Default to a reference date if no temporal data found
        intime = pd.Timestamp('2000-01-01 00:00:00')

    # Build measures dictionary using NEW feature names
    measures = {}
    all_skip_features = EXCLUDE_FEATURES | ADDITIONAL_SKIP_FEATURES

    for feature_name, feature_value in patient_data.items():
        # Skip metadata and excluded features
        if feature_name in all_skip_features:
            continue

        # Handle temporal features (list of dicts)
        if isinstance(feature_value, list):
            temporal_dict = convert_temporal_list_to_dict(feature_value)
            if temporal_dict:
                # Use the NEW feature name directly
                measures[feature_name] = temporal_dict

        # Handle static features (scalar values)
        else:
            # Convert None to 0 for compatibility
            if feature_value is None:
                feature_value = 0

            # Convert boolean to int for numerical processing
            if isinstance(feature_value, bool):
                feature_value = int(feature_value)

            # Keep strings as-is (for categorical features like gender, race)
            # Use the NEW feature name directly
            measures[feature_name] = feature_value

    # Create Patient object
    patient = Patient(
        subject_id=subject_id,
        hadm_id=hadm_id,
        stay_id=stay_id,
        intime=intime,
        akdPositive=akd_positive,
        measures=measures,
        akdTime=Timedelta(days=10)  # Default value
    )

    return patient


def load_new_format_data(filepath: Union[str, Path]) -> Patients:
    """
    Load patient data from joblib file in new format and convert to Patients object.

    Uses all available features from the new format without name mapping.

    Args:
        filepath: Path to joblib file containing dictionary in new format

    Returns:
        Patients object compatible with TXGBoost pipeline

    Example:
        >>> patients = load_new_format_data('data/new_format_patients.joblib')
        >>> print(f"Loaded {len(patients)} patients")
        >>> # Now use with existing pipeline
        >>> from TBoostv2 import main
        >>> # patients can be used directly in the pipeline
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from {filepath}...")

    # Load joblib file
    data_dict = joblib.load(filepath)

    if not isinstance(data_dict, dict):
        raise ValueError(f"Expected dict, got {type(data_dict)}")

    print(f"Loaded {len(data_dict)} patient records")

    # Convert each patient record
    patient_list = []
    conversion_errors = 0
    feature_names_set = set()

    for stay_id_str, patient_data in data_dict.items():
        try:
            patient = convert_patient_data(stay_id_str, patient_data)
            patient_list.append(patient)

            # Track all feature names
            feature_names_set.update(patient.measures.keys())

        except Exception as e:
            print(f"Warning: Failed to convert patient {stay_id_str}: {e}")
            conversion_errors += 1
            continue

    if conversion_errors > 0:
        print(f"Warning: {conversion_errors} patients failed conversion")

    print(f"Successfully converted {len(patient_list)} patients")
    print(f"Total unique features: {len(feature_names_set)}")

    # Categorize features
    temporal_features = []
    static_features = []

    if len(patient_list) > 0:
        sample_patient = patient_list[0]
        for feature_name, feature_value in sample_patient.measures.items():
            if isinstance(feature_value, dict):
                temporal_features.append(feature_name)
            else:
                static_features.append(feature_name)

        print(f"  Temporal features: {len(temporal_features)}")
        print(f"  Static features: {len(static_features)}")

    # Create Patients object
    patients = Patients(patients=patient_list)

    # Print summary statistics
    aki_count = sum([1 for p in patients.patientList if p.akdPositive])
    print(f"  AKI positive: {aki_count} ({aki_count / len(patients):.2%})")
    print(f"  AKI negative: {len(patients) - aki_count} ({(len(patients) - aki_count) / len(patients):.2%})")

    return patients


def load_and_prepare_new_format_patients(
    filepath: Union[str, Path],
    nullable_measures: List[str] = None,
    min_feature_coverage: float = 0.8
) -> Patients:
    """
    Load and prepare patient data with the same preprocessing as the original pipeline.

    This function uses all features from the NEW format - no name mapping to old format.

    Args:
        filepath: Path to joblib file
        nullable_measures: List of measures to fill with 0 if missing (None = use defaults)
        min_feature_coverage: Minimum fraction of patients that must have a feature

    Returns:
        Preprocessed Patients object ready for model training
    """
    from constants import NULLABLE_MEASURES

    if nullable_measures is None:
        # Use default nullable measures, but note these may not match new format exactly
        # The preprocessing will work with whatever features are present
        nullable_measures = NULLABLE_MEASURES

    # Load data
    patients = load_new_format_data(filepath)

    print("\nPreprocessing data...")

    # Fill missing nullable measures (only if they exist in the data)
    actual_nullable = [m for m in nullable_measures if m in patients.getMeasures()]
    if actual_nullable:
        patients.fillMissingMeasureValue(actual_nullable, 0)
        print(f"  Filled {len(actual_nullable)} nullable measures with 0")

    # Remove features with low coverage
    measures = patients.getMeasures()
    features_to_remove = []

    for measure, count in measures.items():
        coverage = count / len(patients)
        if coverage < min_feature_coverage:
            features_to_remove.append(measure)
            print(f"  Removing {measure}: coverage {coverage:.1%} < {min_feature_coverage:.0%}")

    if features_to_remove:
        patients.removeMeasures(features_to_remove)
        print(f"  Removed {len(features_to_remove)} low-coverage features")

    # Remove patients with missing features
    initial_count = len(patients)
    patients.removePatientByMissingFeatures()
    removed_count = initial_count - len(patients)

    if removed_count > 0:
        print(f"  Removed {removed_count} patients with missing features")

    print(f"\nFinal dataset: {len(patients)} patients")

    # Print final feature summary
    final_measures = patients.getMeasures()
    print(f"Final features: {len(final_measures)}")

    # Show feature names
    print("\nAvailable features (first 20):")
    for i, feature_name in enumerate(sorted(list(final_measures.keys())[:20])):
        print(f"  {feature_name}")
    if len(final_measures) > 20:
        print(f"  ... and {len(final_measures) - 20} more")

    return patients


# Example usage and testing
if __name__ == "__main__":
    import sys

    print("="*80)
    print("NEW DATA LOADER TEST")
    print("="*80)
    print("Using NEW format feature names directly (no mapping)")
    print("="*80)

    if len(sys.argv) > 1:
        # Load from command line argument
        filepath = sys.argv[1]
        patients = load_and_prepare_new_format_patients(filepath)

        print("\nSample patient info:")
        if len(patients) > 0:
            sample = patients.patientList[0]
            print(f"  Subject ID: {sample.subject_id}")
            print(f"  Stay ID: {sample.stay_id}")
            print(f"  AKI Positive: {sample.akdPositive}")
            print(f"  Intime: {sample.intime}")
            print(f"  Number of measures: {len(sample.measures)}")

            # Show temporal vs static
            temporal = [k for k, v in sample.measures.items() if isinstance(v, dict)]
            static = [k for k, v in sample.measures.items() if not isinstance(v, dict)]

            print(f"\n  Temporal features ({len(temporal)}):")
            for feat in sorted(temporal)[:10]:
                print(f"    {feat}")

            print(f"\n  Static features ({len(static)}):")
            for feat in sorted(static)[:10]:
                val = sample.measures[feat]
                print(f"    {feat}: {val}")
    else:
        print("\nUsage: python new_data_loader.py <path_to_joblib_file>")
        print("\nOr import in your code:")
        print("  from new_data_loader import load_and_prepare_new_format_patients")
        print("  patients = load_and_prepare_new_format_patients('data.joblib')")
        print("\nNote: This loader uses feature names from the NEW format directly.")
        print("      No mapping to old format names is performed.")
