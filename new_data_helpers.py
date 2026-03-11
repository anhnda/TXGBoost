"""
Helper functions for new data format.

These functions properly distinguish between temporal and static features
in the new data format.
"""

def get_temporal_features_from_new_format(patients):
    """
    Get only TEMPORAL features from new format data.

    Temporal features are those that are stored as dicts with Timestamp keys
    (i.e., they have time-series data).

    Static features (scalars like age, gender, etc.) are excluded.

    Args:
        patients: Patients object

    Returns:
        List of temporal feature names (sorted)
    """
    if len(patients.patientList) == 0:
        return []

    # Sample patient to check feature types
    sample_patient = patients.patientList[0]

    temporal_features = []
    static_features = []

    for feature_name, feature_value in sample_patient.measures.items():
        # Temporal features are stored as dicts (Timestamp -> value)
        if isinstance(feature_value, dict) and len(feature_value) > 0:
            temporal_features.append(feature_name)
        else:
            # Static features are scalars or empty dicts
            static_features.append(feature_name)

    # Verify across multiple patients to be sure
    if len(patients.patientList) > 1:
        for patient in patients.patientList[1:min(10, len(patients.patientList))]:
            for feature_name in list(temporal_features):
                if feature_name in patient.measures:
                    value = patient.measures[feature_name]
                    # If it's ever NOT a dict, it's not temporal
                    if not isinstance(value, dict):
                        if feature_name in temporal_features:
                            temporal_features.remove(feature_name)
                            if feature_name not in static_features:
                                static_features.append(feature_name)

    print(f"\n[Feature Detection]")
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"  Static features: {len(static_features)}")

    if len(temporal_features) > 0:
        print(f"\n  Sample temporal features:")
        for feat in sorted(temporal_features)[:10]:
            # Show how many time points this feature has
            sample_data = sample_patient.measures[feat]
            print(f"    {feat}: {len(sample_data)} time points")
        if len(temporal_features) > 10:
            print(f"    ... and {len(temporal_features) - 10} more")

    if len(static_features) > 0:
        print(f"\n  Sample static features:")
        for feat in sorted(static_features)[:10]:
            sample_value = sample_patient.measures[feat]
            print(f"    {feat}: {sample_value}")
        if len(static_features) > 10:
            print(f"    ... and {len(static_features) - 10} more")

    return sorted(temporal_features)


def get_static_features_from_new_format(patients):
    """
    Get only STATIC features from new format data.

    Static features are those that are stored as scalars (not time-series).

    Args:
        patients: Patients object

    Returns:
        List of static feature names (sorted)
    """
    if len(patients.patientList) == 0:
        return []

    sample_patient = patients.patientList[0]

    static_features = []

    for feature_name, feature_value in sample_patient.measures.items():
        # Static features are NOT dicts (they're scalars)
        if not isinstance(feature_value, dict):
            static_features.append(feature_name)
        elif isinstance(feature_value, dict) and len(feature_value) == 0:
            # Empty dicts are also static (no time points)
            static_features.append(feature_name)

    return sorted(static_features)


def validate_temporal_features(patients, feature_names):
    """
    Validate that the given features are actually temporal.

    Args:
        patients: Patients object
        feature_names: List of feature names to validate

    Returns:
        Tuple of (valid_features, invalid_features)
    """
    if len(patients.patientList) == 0:
        return [], feature_names

    valid = []
    invalid = []

    # Check first few patients
    check_patients = patients.patientList[:min(10, len(patients.patientList))]

    for feature_name in feature_names:
        is_temporal = True

        for patient in check_patients:
            if feature_name in patient.measures:
                value = patient.measures[feature_name]
                # Temporal features must be dicts with data
                if not isinstance(value, dict) or len(value) == 0:
                    is_temporal = False
                    break

        if is_temporal:
            valid.append(feature_name)
        else:
            invalid.append(feature_name)

    if len(invalid) > 0:
        print(f"\n[Warning] Found {len(invalid)} non-temporal features:")
        for feat in invalid[:10]:
            print(f"  - {feat}")
        if len(invalid) > 10:
            print(f"  ... and {len(invalid) - 10} more")

    return valid, invalid
