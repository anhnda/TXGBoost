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
import torch.optim
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
    # FIXED_FEATURES,  # Not used - we detect features dynamically from new format
    SimpleStaticEncoder,
    # GatedDecisionHead,  # Imported in train_rnn_extractor_new_format
    EnhancedHybridDataset,
    enhanced_collate_fn,
    RNNFeatureExtractor,
    # train_rnn_extractor,  # Not used - we use train_rnn_extractor_new_format
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

# Import helpers for new data format
from new_data_helpers import (
    get_temporal_features_from_new_format,
    get_static_features_from_new_format,
    validate_temporal_features,
)

import torch.nn as nn
from TimeEmbedding import DEVICE


# ==============================================================================
# Soft Decision Tree - Differentiable Decision Tree for RNN Pre-training
# ==============================================================================

class SoftDecisionTree(nn.Module):
    """
    Soft Decision Tree (SDT) - A fully differentiable decision tree.

    Key advantages over GatedDecisionHead:
    - Explicit tree structure mimics XGBoost's decision trees
    - Soft routing allows gradients to flow through all paths
    - Hierarchical feature interactions through tree levels
    - Stronger inductive bias for tabular data

    Architecture:
    - Internal nodes: Linear transformation + Sigmoid (soft split)
    - Leaf nodes: Learned output values
    - Forward pass: Soft routing probability × leaf values
    """

    def __init__(self, input_dim, depth=5, dropout=0.2):
        """
        Args:
            input_dim: Input feature dimension
            depth: Tree depth (depth=5 → 32 leaves, depth=6 → 64 leaves)
            dropout: Dropout rate for regularization
        """
        super(SoftDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_internal_nodes = 2 ** depth - 1

        # Feature normalization for better training stability
        self.feature_norm = nn.LayerNorm(input_dim)

        # Internal nodes: Each node learns a linear decision function
        # Node i decides: sigmoid(W_i @ x + b_i) → left vs right
        self.internal_weights = nn.Parameter(
            torch.randn(self.num_internal_nodes, input_dim) * 0.1
        )
        self.internal_biases = nn.Parameter(
            torch.zeros(self.num_internal_nodes)
        )

        # Leaf nodes: Each leaf has a learned output value
        self.leaf_values = nn.Parameter(
            torch.randn(self.num_leaves) * 0.1
        )

        # Temperature for soft routing (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 5.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            output: [batch_size, 1] predictions in [0, 1]
        """
        batch_size = x.size(0)

        # Normalize input features
        x = self.feature_norm(x)
        x = self.dropout(x)

        # Calculate routing probabilities for all internal nodes
        # node_logits: [batch_size, num_internal_nodes]
        node_logits = torch.matmul(x, self.internal_weights.t()) + self.internal_biases
        node_probs = torch.sigmoid(node_logits * torch.abs(self.temperature))

        # Calculate probability of reaching each leaf via soft routing
        # leaf_probs: [batch_size, num_leaves]
        leaf_probs = self._compute_leaf_probabilities(node_probs)

        # Weighted combination of leaf values
        # output: [batch_size]
        output = torch.matmul(leaf_probs, self.leaf_values)

        # Apply sigmoid to get final probability
        return torch.sigmoid(output).unsqueeze(-1)

    def _compute_leaf_probabilities(self, node_probs):
        """
        Compute probability of reaching each leaf through soft routing.

        For a binary tree:
        - Node i's left child: 2*i + 1
        - Node i's right child: 2*i + 2
        - P(reach leaf j) = product of routing decisions along path

        Args:
            node_probs: [batch_size, num_internal_nodes] - prob of going RIGHT at each node
        Returns:
            leaf_probs: [batch_size, num_leaves]
        """
        batch_size = node_probs.size(0)
        leaf_probs = torch.ones(batch_size, self.num_leaves, device=node_probs.device)

        # For each leaf, trace path from root and accumulate probabilities
        for leaf_idx in range(self.num_leaves):
            path = self._get_path_to_leaf(leaf_idx)

            for node_idx, direction in path:
                if direction == 'right':
                    # Going right: use node probability
                    leaf_probs[:, leaf_idx] *= node_probs[:, node_idx]
                else:
                    # Going left: use 1 - node probability
                    leaf_probs[:, leaf_idx] *= (1 - node_probs[:, node_idx])

        return leaf_probs

    def _get_path_to_leaf(self, leaf_idx):
        """
        Get the path from root to a specific leaf.

        Returns:
            path: List of (node_idx, direction) tuples
        """
        # Leaf indexing: leaf_idx ranges from 0 to num_leaves-1
        # Internal nodes: 0 to num_internal_nodes-1
        # Convert leaf_idx to position in complete binary tree
        node_idx = leaf_idx + self.num_internal_nodes

        path = []
        while node_idx > 0:
            parent_idx = (node_idx - 1) // 2
            if node_idx % 2 == 1:
                direction = 'left'
            else:
                direction = 'right'
            path.append((parent_idx, direction))
            node_idx = parent_idx

        return list(reversed(path))


class AdaptiveNeuralTree(nn.Module):
    """
    Adaptive Neural Tree - Enhanced decision tree with adaptive depth and attention.

    Improvements over basic SoftDecisionTree:
    - Feature attention mechanism for better feature selection
    - Adaptive routing with confidence scores
    - Ensemble of multiple trees for robustness
    """

    def __init__(self, input_dim, depth=5, num_trees=3, dropout=0.2):
        super(AdaptiveNeuralTree, self).__init__()
        self.num_trees = num_trees

        # Feature attention: Learn which features are important
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

        # Ensemble of soft decision trees
        self.trees = nn.ModuleList([
            SoftDecisionTree(input_dim, depth=depth, dropout=dropout)
            for _ in range(num_trees)
        ])

        # Tree weighting: Learn to weight different trees
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            output: [batch_size, 1] predictions
        """
        # Apply feature attention
        attention = self.feature_attention(x)
        x_attended = x * attention

        # Get predictions from all trees
        tree_outputs = []
        for tree in self.trees:
            tree_out = tree(x_attended)
            tree_outputs.append(tree_out)

        # Weighted ensemble
        tree_outputs = torch.cat(tree_outputs, dim=-1)  # [batch_size, num_trees]
        weights = torch.softmax(self.tree_weights, dim=0)
        output = torch.matmul(tree_outputs, weights).unsqueeze(-1)

        return output


def train_rnn_extractor_new_format(model, train_loader, val_loader, criterion, optimizer,
                                     static_dim, epochs=50):
    """
    Train RNN extractor with correct dimensions for new data format.

    This is a wrapper around the original train_rnn_extractor that properly
    handles the dynamic static dimension from new format data.
    """
    rnn_dim = model.rnn_cell.hidden_dim

    # Import here to avoid circular dependency
    import copy

    # Calculate class weights for imbalanced data
    all_labels = []
    for _, labels, _ in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    pos_count = np.sum(all_labels == 1)
    neg_count = np.sum(all_labels == 0)
    pos_weight = neg_count / (pos_count + 1e-6)

    print(f"  [Stage 1] Class distribution: Pos={pos_count}, Neg={neg_count}, Pos_weight={pos_weight:.2f}")

    # Create STRONGER decision head: AdaptiveNeuralTree (ensemble of soft decision trees)
    # This is much more powerful than GatedDecisionHead because:
    # - Mimics XGBoost's tree structure explicitly
    # - Uses feature attention for better feature selection
    # - Ensemble of 3 trees for robustness
    # - Depth 6 → 64 leaves per tree → high capacity
    temp_head = AdaptiveNeuralTree(
        input_dim=rnn_dim + static_dim,
        depth=6,  # 64 leaves per tree
        num_trees=3,  # Ensemble of 3 trees
        dropout=0.2
    ).to(DEVICE)

    print(f"  [Stage 1] Using AdaptiveNeuralTree: 3 trees × 64 leaves = 192 decision paths")

    full_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(temp_head.parameters()),
        lr=0.001,
        weight_decay=1e-5  # L2 regularization
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=epochs, eta_min=1e-5)

    best_auc = 0
    best_state = None
    best_epoch = 0
    patience = 8  # 8 eval intervals * 5 epochs = 40 epochs max wait
    counter = 0

    print(f"  [Stage 1] Pre-training RNN with AdaptiveNeuralTree (Differentiable Decision Forest)")
    print(f"    RNN dim: {rnn_dim}, Static dim: {static_dim}, Total: {rnn_dim + static_dim}")
    print(f"    Architecture: Feature Attention → 3 Soft Decision Trees (depth 6)")
    print(f"    Early stopping: patience={patience}")
    print(f"    Learning rate: 0.001 -> 1e-5 (cosine annealing)")

    for epoch in range(epochs):
        model.train()
        temp_head.train()

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            h = model(t_data)
            combined = torch.cat([h, s_data], dim=1)
            preds = temp_head(combined).squeeze(-1)

            # Weighted BCE loss for class imbalance
            weights = torch.where(labels == 1, pos_weight, 1.0).to(DEVICE)
            loss = nn.BCELoss(weight=weights)(preds, labels)

            full_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(temp_head.parameters()),
                max_norm=1.0
            )

            full_optimizer.step()

        # Step the learning rate scheduler
        scheduler.step()

        if (epoch+1) % 5 == 0:
            model.eval()
            temp_head.eval()
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    h = model(t_data)
                    combined = torch.cat([h, s_data], dim=1)
                    preds = temp_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_lbls.extend(labels.cpu().numpy())

            from sklearn.metrics import average_precision_score, roc_auc_score
            aupr = average_precision_score(all_lbls, all_preds)
            auc_val = aupr
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1} Val AUPR: {auc_val:.4f} | LR: {current_lr:.6f}")

            if auc_val > best_auc:
                best_auc = auc_val
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    model.load_state_dict(best_state)
    return model


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
    # STEP 2: Extract features and encode (using NEW format detection)
    # ========================================================================
    print("\n[Step 2] Extracting temporal and static features...")

    # Use new format-specific feature detection
    temporal_feats = get_temporal_features_from_new_format(patients)
    static_feats = get_static_features_from_new_format(patients)

    # Validate temporal features
    temporal_feats, invalid_feats = validate_temporal_features(patients, temporal_feats)

    if len(invalid_feats) > 0:
        print(f"\n  Excluded {len(invalid_feats)} non-temporal features")

    print(f"\n  Final temporal features: {len(temporal_feats)}")
    print(f"  Final static features: {len(static_feats)}")

    # Encode static features (use detected static features, not FIXED_FEATURES)
    print("\n[Step 3] Encoding static features...")
    encoder = SimpleStaticEncoder(static_feats)
    encoder.fit(patients.patientList)

    print(f"\n  Feature Summary:")
    print(f"    Temporal Features: {len(temporal_feats)}")
    print(f"    Static Features: {len(static_feats)}")
    print(f"    Global Stats: {len(temporal_feats) * 5} (mean/max/min/std/slope)")
    print(f"    Total Enhanced Static: {len(static_feats) + len(temporal_feats) * 5} dims")

    # ========================================================================
    # STEP 3: Training with cross-validation
    # ========================================================================
    print("\n[Step 4] Starting cross-validation training...")

    # Calculate static dimension for RNN training
    # Enhanced static = len(static_feats) + (len(temporal_feats) * 5 global stats)
    actual_static_dim = len(static_feats) + (len(temporal_feats) * 5)

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

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=enhanced_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=enhanced_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=enhanced_collate_fn)

        # Stage 1: Train RNN
        print("\n  [Stage 1] Training RNN extractor...")
        print(f"    Static features: {len(static_feats)}")
        print(f"    Global stats: {len(temporal_feats) * 5}")
        print(f"    Total static dim: {actual_static_dim}")

        rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=256).to(DEVICE)
        opt = torch.optim.Adam(rnn.parameters(), lr=0.001)

        # Use new format-aware training function
        rnn = train_rnn_extractor_new_format(
            rnn, train_loader, val_loader, nn.BCELoss(), opt,
            static_dim=actual_static_dim,
            epochs=50
        )

        # Stage 2: Extract features
        print("\n  [Stage 2] Extracting triple features...")
        X_train, y_train = get_triple_features(rnn, train_loader)
        X_val, y_val = get_triple_features(rnn, val_loader)
        X_test, y_test = get_triple_features(rnn, test_loader)

        expected_dims = len(temporal_feats) + actual_static_dim + 256
        print(f"    Enhanced feature dimensions: {X_train.shape[1]} (expected: {expected_dims})")
        print(f"      = {len(temporal_feats)} Last + {actual_static_dim} Enhanced Static + 256 RNN")
        print(f"    Enhanced samples: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

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

        print(f"    Baseline feature dimensions: {X_tr_b.shape[1]} (standard features without RNN)")
        print(f"    Baseline samples: Train={len(y_tr_b)}, Test={len(y_te_b)}")

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
