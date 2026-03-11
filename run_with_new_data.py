"""
TBoostv2 Pipeline with New Data Format

This script demonstrates how to use the new data loader with the existing TBoostv2 pipeline.
Simply replace the data loading step with the new format loader.
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
import os

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)

# Set up paths
PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

# Import existing components
from TBoostv2 import (
    seed_everything,
    FIXED_FEATURES,
    SimpleStaticEncoder,
    GatedDecisionHead,
    EnhancedHybridDataset,
    enhanced_collate_fn,
    RNNFeatureExtractor,
    train_rnn_extractor,
    get_triple_features,
    xseed,
)

from TimeEmbeddingVal import (
    get_all_temporal_features,
    split_patients_train_val,
)

from utils.prepare_data import trainTestPatients, encodeCategoricalData

# Import the new data loader
from new_data_loader import load_and_prepare_new_format_patients


def main(data_filepath):
    """
    Run TBoostv2 pipeline with new data format.

    Args:
        data_filepath: Path to joblib file containing data in new format
    """
    print("="*80)
    print("TBoostv2 WITH NEW DATA FORMAT")
    print("="*80)

    # Set random seed for reproducibility
    seed_everything(xseed)

    # ========================================================================
    # STEP 1: Load data using new data loader (with caching)
    # ========================================================================
    print("\n[Step 1] Loading data in new format...")
    print("Note: Using cache for faster loading (cache: tmp/cache_newdata.cache)")
    patients = load_and_prepare_new_format_patients(
        data_filepath,
        min_feature_coverage=0.8,
        use_cache=True  # Enable caching for faster subsequent runs
    )

    print(f"\nLoaded {len(patients)} patients")
    aki_count = sum([1 for p in patients.patientList if p.akdPositive])
    print(f"  AKI positive: {aki_count} ({aki_count / len(patients):.2%})")
    print(f"  AKI negative: {len(patients) - aki_count}")

    # ========================================================================
    # STEP 2: Extract features and encode
    # ========================================================================
    print("\n[Step 2] Extracting temporal and static features...")
    temporal_feats = get_all_temporal_features(patients)
    print(f"  Temporal features: {len(temporal_feats)}")

    # Print available features
    print(f"\n  Available temporal features:")
    for i, feat in enumerate(temporal_feats[:10]):
        print(f"    {feat}")
    if len(temporal_feats) > 10:
        print(f"    ... and {len(temporal_feats) - 10} more")

    # Encode static features
    print("\n[Step 3] Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"  Input: {len(temporal_feats)} Temporal Features")
    print(f"         {len(FIXED_FEATURES)} Static Features")
    print(f"         {len(temporal_feats) * 5} Global Stats (mean/max/min/std/slope)")
    print(f"         Total Enhanced Static: {len(FIXED_FEATURES) + len(temporal_feats) * 5} dims")

    # ========================================================================
    # STEP 3: Training with cross-validation
    # ========================================================================
    print("\n[Step 4] Starting cross-validation training...")

    # Storage for metrics
    metrics_hybrid = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}
    metrics_base = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Cross-validation loop
    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")

        # Split train into train/val
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p, val_p, test_p_list = train_p_obj.patientList, val_p_obj.patientList, test_p.patientList

        print(f"  Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p_list)}")

        # Create datasets
        print("\n  Creating enhanced datasets...")
        train_ds = EnhancedHybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = EnhancedHybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = EnhancedHybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=enhanced_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=enhanced_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=enhanced_collate_fn)

        # Stage 1: Train RNN
        print("\n  [Stage 1] Training RNN extractor...")
        from TimeEmbedding import DEVICE
        rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=128).to(DEVICE)
        opt = torch.optim.Adam(rnn.parameters(), lr=0.001)
        rnn = train_rnn_extractor(rnn, train_loader, val_loader, nn.BCELoss(), opt, epochs=50)

        # Stage 2: Extract features
        print("\n  [Stage 2] Extracting triple features...")
        X_train, y_train = get_triple_features(rnn, train_loader)
        X_val, y_val = get_triple_features(rnn, val_loader)
        X_test, y_test = get_triple_features(rnn, test_loader)

        print(f"    Feature dimensions: {X_train.shape[1]} (25 Last + 148 Enhanced Static + 128 RNN)")

        # Stage 3: Train XGBoost
        print("\n  [Stage 3] Training XGBoost...")
        ratio = np.sum(y_train==0) / (np.sum(y_train==1) + 1e-6)

        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=ratio, eval_metric='auc', random_state=42
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate hybrid model
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        metrics_hybrid['auc'].append(roc_auc_score(y_test, y_prob))
        metrics_hybrid['acc'].append(accuracy_score(y_test, y_pred))
        metrics_hybrid['spec'].append(tn / (tn + fp))
        metrics_hybrid['prec'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics_hybrid['rec'].append(recall_score(y_test, y_pred))
        metrics_hybrid['auc_pr'].append(auc(rec, prec))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {metrics_hybrid['auc'][-1]:.3f})")

        # Baseline model
        print("\n  [Baseline] Training standard XGBoost...")
        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]

        xgb_base = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=ratio, eval_metric='auc', random_state=42
        )
        xgb_base.fit(X_tr_b, y_tr_b)

        y_prob_b = xgb_base.predict_proba(X_te_b)[:, 1]
        y_pred_b = (y_prob_b > 0.5).astype(int)

        tn, fp, _, _ = confusion_matrix(y_te_b, y_pred_b).ravel()
        prec_b, rec_b, _ = precision_recall_curve(y_te_b, y_prob_b)

        metrics_base['auc'].append(roc_auc_score(y_te_b, y_prob_b))
        metrics_base['acc'].append(accuracy_score(y_te_b, y_pred_b))
        metrics_base['spec'].append(tn / (tn + fp))
        metrics_base['prec'].append(precision_score(y_te_b, y_pred_b, zero_division=0))
        metrics_base['rec'].append(recall_score(y_te_b, y_pred_b))
        metrics_base['auc_pr'].append(auc(rec_b, prec_b))

        fpr_b, tpr_b, _ = roc_curve(y_te_b, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {metrics_base['auc'][-1]:.3f})")

        print(f"\n  Fold {fold} Results -> Enhanced: {metrics_hybrid['auc'][-1]:.3f} vs Baseline: {metrics_base['auc'][-1]:.3f}")

    # ========================================================================
    # STEP 4: Results and visualization
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    # Plot configuration
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

    ax1.set_title("Enhanced TBoostv2 (New Data Format)")
    ax2.set_title("Baseline (New Data Format)")

    plt.tight_layout()
    plt.savefig("result/tboostv2_new_data_vs_baseline.png", dpi=300)
    print("\nPlot saved to: result/tboostv2_new_data_vs_baseline.png")

    # Print statistics
    def print_stat(name, h_metrics, b_metrics):
        h_mean, h_std = np.mean(h_metrics), np.std(h_metrics)
        b_mean, b_std = np.mean(b_metrics), np.std(b_metrics)
        improvement = ((h_mean - b_mean) / b_mean) * 100
        print(f"{name:15s} | Enhanced: {h_mean:.4f} ± {h_std:.4f}  |  "
              f"Baseline: {b_mean:.4f} ± {b_std:.4f}  |  "
              f"Improvement: {improvement:+.2f}%")

    print("\nMetric Comparison:")
    print("-" * 100)
    print_stat("AUC", metrics_hybrid['auc'], metrics_base['auc'])
    print_stat("AUC-PR", metrics_hybrid['auc_pr'], metrics_base['auc_pr'])
    print_stat("Accuracy", metrics_hybrid['acc'], metrics_base['acc'])
    print_stat("Specificity", metrics_hybrid['spec'], metrics_base['spec'])
    print_stat("Precision", metrics_hybrid['prec'], metrics_base['prec'])
    print_stat("Recall", metrics_hybrid['rec'], metrics_base['rec'])

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

NEW_DATA_PATH = "new_data/final_dataset.pkl"
if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("\nUsage: python run_with_new_data.py <path_to_joblib_file>")
    #     print("\nExample:")
    #     print("  python run_with_new_data.py data/patients_new_format.joblib")
    #     sys.exit(1)

    data_filepath = NEW_DATA_PATH

    if not os.path.exists(data_filepath):
        print(f"\nError: File not found: {data_filepath}")
        sys.exit(1)

    main(data_filepath)
