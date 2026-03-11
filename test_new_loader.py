"""
Test script for new data loader

This script demonstrates the data conversion process using example files.
"""

import json
import joblib
from pathlib import Path
from new_data_loader import (
    convert_patient_data,
    convert_temporal_list_to_dict,
    load_new_format_data,
)


def test_temporal_conversion():
    """Test temporal data conversion"""
    print("="*80)
    print("TEST 1: Temporal Data Conversion")
    print("="*80)

    # Example temporal data in new format
    temporal_list = [
        {'charttime': '2162-06-21 05:46:00', 'value': 74.0},
        {'charttime': '2162-06-21 06:00:00', 'value': 74.0},
        {'charttime': '2162-06-21 07:00:00', 'value': 66.0},
    ]

    print("\nInput (new format):")
    for item in temporal_list[:3]:
        print(f"  {item}")

    # Convert
    temporal_dict = convert_temporal_list_to_dict(temporal_list)

    print("\nOutput (old format):")
    for ts, val in list(temporal_dict.items())[:3]:
        print(f"  {ts}: {val}")

    print(f"\n✓ Converted {len(temporal_dict)} time points")


def test_patient_conversion():
    """Test single patient conversion"""
    print("\n" + "="*80)
    print("TEST 2: Patient Record Conversion")
    print("="*80)

    # Load example new format data
    example_file = Path("example/New_example_bio.txt")

    if not example_file.exists():
        print("⚠ Example file not found, skipping test")
        return

    # Parse the example file (it's a dict printed as text)
    with open(example_file, 'r') as f:
        content = f.read()
        # This is a printed dict, so we need to eval it (normally use joblib.load)
        try:
            data_dict = eval(content)
        except:
            print("⚠ Could not parse example file")
            return

    # Get first patient
    stay_id = list(data_dict.keys())[0]
    patient_data = data_dict[stay_id]

    print(f"\nStay ID: {stay_id}")
    print(f"Target: {patient_data.get('target')}")
    print(f"Age: {patient_data.get('age_at_admission')}")
    print(f"Gender: {patient_data.get('gender')}")

    # Count temporal vs static features
    temporal_count = 0
    static_count = 0

    for key, value in patient_data.items():
        if isinstance(value, list):
            temporal_count += 1
        else:
            static_count += 1

    print(f"\nFeatures in new format:")
    print(f"  Temporal: {temporal_count}")
    print(f"  Static: {static_count}")
    print(f"  Total: {temporal_count + static_count}")

    # Convert to Patient object
    print("\nConverting to Patient object...")
    patient = convert_patient_data(stay_id, patient_data)

    print(f"\n✓ Conversion successful!")
    print(f"  Subject ID: {patient.subject_id}")
    print(f"  Stay ID: {patient.stay_id}")
    print(f"  AKI Positive: {patient.akdPositive}")
    print(f"  Intime: {patient.intime}")
    print(f"  Number of measures: {len(patient.measures)}")

    # Show some converted measures
    print("\nSample converted measures:")
    for i, (key, value) in enumerate(list(patient.measures.items())[:5]):
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} time points")
        else:
            print(f"  {key}: {value}")

    print(f"\n✓ Patient object ready for pipeline!")


def create_test_joblib():
    """Create a small test joblib file from the example"""
    print("\n" + "="*80)
    print("TEST 3: Creating Test Joblib File")
    print("="*80)

    example_file = Path("example/New_example_bio.txt")

    if not example_file.exists():
        print("⚠ Example file not found, skipping test")
        return None

    # Parse example
    with open(example_file, 'r') as f:
        content = f.read()
        try:
            data_dict = eval(content)
        except:
            print("⚠ Could not parse example file")
            return None

    # Save as joblib
    output_file = Path("example/test_patients.joblib")
    joblib.dump(data_dict, output_file)

    print(f"\n✓ Created test file: {output_file}")
    print(f"  Patients: {len(data_dict)}")

    return output_file


def test_full_load():
    """Test full loading pipeline"""
    print("\n" + "="*80)
    print("TEST 4: Full Loading Pipeline")
    print("="*80)

    # Create test file
    test_file = create_test_joblib()

    if test_file is None:
        print("⚠ Cannot create test file, skipping")
        return

    # Load using the full loader
    print(f"\nLoading data from {test_file}...")
    patients = load_new_format_data(test_file)

    print(f"\n✓ Loading successful!")
    print(f"  Total patients: {len(patients)}")

    # Count AKI positive/negative
    aki_positive = sum([1 for p in patients.patientList if p.akdPositive])
    aki_negative = len(patients) - aki_positive

    print(f"  AKI positive: {aki_positive}")
    print(f"  AKI negative: {aki_negative}")

    # Check measures
    if len(patients) > 0:
        sample = patients.patientList[0]
        print(f"\nSample patient:")
        print(f"  ID: {sample.stay_id}")
        print(f"  Measures: {len(sample.measures)}")

        # List temporal measures
        temporal_measures = [k for k, v in sample.measures.items() if isinstance(v, dict)]
        static_measures = [k for k, v in sample.measures.items() if not isinstance(v, dict)]

        print(f"  Temporal: {len(temporal_measures)}")
        print(f"    Examples: {temporal_measures[:5]}")
        print(f"  Static: {len(static_measures)}")
        print(f"    Examples: {static_measures[:5]}")

    print("\n✓ Data ready for TXGBoost pipeline!")


def compare_formats():
    """Compare old and new formats side by side"""
    print("\n" + "="*80)
    print("TEST 5: Format Comparison")
    print("="*80)

    old_file = Path("example/Old_example_bio.txt")
    new_file = Path("example/New_example_bio.txt")

    if not old_file.exists() or not new_file.exists():
        print("⚠ Example files not found")
        return

    print("\nOLD FORMAT:")
    print("  Structure: List of dicts")
    print("  Keys: subjectId, hadmId, stayId, akdPositive, measures")
    print("  Temporal: {timestamp_str: value}")

    print("\nNEW FORMAT:")
    print("  Structure: Dict with stay_id keys")
    print("  Keys: stay_id (key), target, feature names")
    print("  Temporal: [{'charttime': ..., 'value': ...}]")

    print("\nCONVERSION MAPPING:")
    print("  subjectId → derived from stay_id")
    print("  hadmId → derived from stay_id")
    print("  stayId → dictionary key")
    print("  akdPositive → target")
    print("  hr → heart_rate")
    print("  scr → creatinine")
    print("  bg → glucose")
    print("  ... (see FEATURE_NAME_MAPPING)")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "NEW DATA LOADER TEST SUITE" + " "*32 + "║")
    print("╚" + "="*78 + "╝")

    try:
        test_temporal_conversion()
        test_patient_conversion()
        test_full_load()
        compare_formats()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Prepare your data in joblib format")
        print("  2. Run: python run_with_new_data.py <your_data.joblib>")
        print("  3. Check results in result/ folder")
        print("\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
