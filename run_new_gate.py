"""
TBoostv2 Pipeline with New Data Format using PyTorch Tabular GATE

This script demonstrates how to use the new data loader with the existing TBoostv2 pipeline
but with PyTorch Tabular's Gated Additive Tree Ensemble (GATE) instead of XGBoost/CatBoost.
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
import os

from pytorch_tabular import TabularModel
from pytorch_tabular.models import GatedAdditiveTreeEnsembleConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

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
from TBoostv3 import (
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
import torch.nn.functional as F
from TimeEmbedding import DEVICE


# ==============================================================================
# Loss Functions
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing - Addresses class imbalance and overfitting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
    - alpha: Balances positive/negative examples
    - gamma: Focuses on hard examples (typical: 2.0)
    - (1 - p_t)^gamma: Down-weights easy examples
    - label_smoothing: Prevents overconfidence (0.1 = smooth 1→0.9, 0→0.1)

    Much better than weighted BCE for severe imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size] raw predictions (no sigmoid)
            targets: [batch_size] ground truth (0 or 1)
        """
        # Apply label smoothing
        # 1 → 1 - smoothing, 0 → smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        # Get probabilities
        probs = torch.sigmoid(logits)

        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ==============================================================================
# Improved Decision Heads for RNN Pre-training
# ==============================================================================

class ImprovedGatedHead(nn.Module):
    """
    Simplified Gated Head - Designed to prevent overfitting.

    Key anti-overfitting measures:
    - Simpler architecture (removed wide path - too much capacity)
    - Much higher dropout (0.5-0.6)
    - Feature dropout layer
    - Fewer parameters overall
    """

    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super(ImprovedGatedHead, self).__init__()

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Feature dropout - randomly drop features during training
        self.feature_dropout = nn.Dropout(dropout * 0.4)

        # Feature selection gate (learns which features matter)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(),
            nn.Dropout(dropout * 0.6)
        )

        # Single deep path (removed wide path to reduce capacity)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8)
        )

        # Final prediction (outputs logits)
        self.output = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)

        # Feature dropout (stronger regularization)
        x = self.feature_dropout(x)

        # Apply learned feature gate
        gate = self.gate(x)
        x_gated = x * gate

        # Encode features
        features = self.encoder(x_gated)

        # Output logits (no sigmoid - using FocalLoss)
        return self.output(features)

    def initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SoftDecisionTree(nn.Module):
    """
    Soft Decision Tree (SDT) - A fully differentiable decision tree.

    Key advantages over GatedDecisionHead:
    - Explicit tree structure mimics GATE's decision trees
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

        # Return logits (no sigmoid - will use BCEWithLogitsLoss)
        return output.unsqueeze(-1)

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

    pos_ratio = pos_count / (pos_count + neg_count)
    print(f"  [Stage 1] Class distribution: Pos={pos_count} ({pos_ratio:.1%}), Neg={neg_count}, Pos_weight={pos_weight:.2f}")

    # Create Focal Loss with Label Smoothing
    # alpha = weight for positive class (higher = focus more on positives)
    # gamma = focusing parameter (higher = focus more on hard examples)
    # label_smoothing = prevents overconfidence (critical for overfitting)
    focal_alpha = min(0.75, pos_ratio * 4)  # Scale with imbalance, cap at 0.75
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0, label_smoothing=0.15)
    print(f"  [Stage 1] Using Focal Loss: alpha={focal_alpha:.3f}, gamma=2.0, label_smoothing=0.15")

    # Create decision head: Simplified ImprovedGatedHead
    # Anti-overfitting design:
    # - Simpler architecture (single path, not multi-scale)
    # - Very high dropout (0.5) at all layers
    # - Feature dropout layer
    # - Label smoothing in loss
    temp_head = ImprovedGatedHead(
        input_dim=rnn_dim + static_dim,
        hidden_dim=192,  # Reduced from 256 to prevent overfitting
        dropout=0.5  # Much higher dropout
    ).to(DEVICE)

    # Initialize weights for better convergence
    temp_head.initialize_weights()

    print(f"  [Stage 1] Using Simplified ImprovedGatedHead: Single-path + Heavy Dropout (0.5)")

    full_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(temp_head.parameters()),
        lr=0.0005,  # Lower LR to prevent rapid overfitting
        weight_decay=1e-4  # Stronger weight decay
    )

    # Add learning rate scheduler
    # ReduceLROnPlateau - only reduce when stuck
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        full_optimizer, mode='max', factor=0.5, patience=3,
        min_lr=1e-6, verbose=True
    )

    # Mixed precision training for better GPU utilization
    scaler = torch.amp.GradScaler('cuda')

    best_auc = 0
    best_state = None
    best_epoch = 0
    best_train_loss = float('inf')
    patience = 6  # Reduced patience - stop earlier if not improving
    counter = 0
    overfitting_counter = 0  # Track consecutive epochs of overfitting

    print(f"  [Stage 1] Pre-training RNN with Simplified ImprovedGatedHead + Focal Loss")
    print(f"    RNN dim: {rnn_dim}, Static dim: {static_dim}, Total: {rnn_dim + static_dim}")
    print(f"    Architecture: BatchNorm → Feature Dropout → Gate → Deep Encoder")
    print(f"    Loss: Focal Loss + Label Smoothing (0.15)")
    print(f"    Anti-overfitting: Dropout=0.5, Feature Dropout, Weight Decay=1e-4")
    print(f"    Early stopping: patience={patience} (stops on val plateau OR overfitting)")
    print(f"    Learning rate: 0.0005 (ReduceLROnPlateau, patience=3, factor=0.5)")
    print(f"    Mixed precision: Enabled (FP16/FP32 automatic)")

    for epoch in range(epochs):
        model.train()
        temp_head.train()
        epoch_loss = 0.0
        num_batches = 0

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)

            full_optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                h = model(t_data)
                combined = torch.cat([h, s_data], dim=1)
                logits = temp_head(combined).squeeze(-1)

                # Focal loss for severe class imbalance
                loss = criterion(logits, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate norm)
            scaler.unscale_(full_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(temp_head.parameters()),
                max_norm=1.0
            )

            # Optimizer step with scaler
            scaler.step(full_optimizer)
            scaler.update()

            # Track loss
            epoch_loss += loss.item()
            num_batches += 1

        # Log training loss every epoch
        avg_loss = epoch_loss / num_batches
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1} Train Loss: {avg_loss:.4f}", end=" | ")

        if (epoch+1) % 5 == 0:
            model.eval()
            temp_head.eval()
            all_preds, all_lbls = [], []
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    h = model(t_data)
                    combined = torch.cat([h, s_data], dim=1)
                    logits = temp_head(combined).squeeze(-1)
                    # Apply sigmoid to convert logits to probabilities
                    preds = torch.sigmoid(logits)
                    all_preds.extend(preds.cpu().numpy())
                    all_lbls.extend(labels.cpu().numpy())

            from sklearn.metrics import average_precision_score, roc_auc_score
            aupr = average_precision_score(all_lbls, all_preds)
            auc_val = aupr

            # Prediction distribution analysis
            pred_mean = np.mean(all_preds)
            pred_std = np.std(all_preds)
            pred_pos_rate = np.sum(np.array(all_preds) > 0.5) / len(all_preds)
            true_pos_rate = np.mean(all_lbls)

            # Step scheduler with validation metric (ReduceLROnPlateau)
            scheduler.step(auc_val)

            current_lr = full_optimizer.param_groups[0]['lr']
            print(f"Val AUPR: {auc_val:.4f} | LR: {current_lr:.6f} | Pred: {pred_mean:.3f}±{pred_std:.3f} | "
                  f"PredPos: {pred_pos_rate:.1%} vs TruePos: {true_pos_rate:.1%}")

            # Detect severe overfitting (train loss very low but val not improving)
            if avg_loss < 0.15 and auc_val < best_auc - 0.02:
                overfitting_counter += 1
                if overfitting_counter >= 2:
                    print(f"    ⚠ Severe overfitting detected (train={avg_loss:.4f}, best val={best_auc:.4f}). Stopping early.")
                    break
            else:
                overfitting_counter = 0

            if auc_val > best_auc:
                best_auc = auc_val
                best_state = copy.deepcopy(model.state_dict())
                best_train_loss = avg_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1} (val AUPR not improving)")
                    break

    model.load_state_dict(best_state)
    return model


def main(data_filepath):
    """
    Run TBoostv2 pipeline with new data format using PyTorch Tabular GATE.

    Args:
        data_filepath: Path to joblib file containing data in new format
    """
    print("="*80)
    print("TBoostv3 WITH NEW DATA FORMAT (PyTorch Tabular GATE)")
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

        # ====================================================================
        # OPTIMIZED DATALOADER CONFIGURATION
        # ====================================================================
        # Balance between:
        # - Large batches (GPU efficiency) vs Small batches (gradient quality)
        # - Batch=128: Good compromise for both GPU utilization & learning
        # - num_workers: 4 for parallel loading without overhead
        # - pin_memory: Faster CPU→GPU transfer
        # - persistent_workers: Keeps workers alive
        # ====================================================================
        print(f"\n  DataLoader: Batch=128, Workers=4, AMP=Enabled, pin_memory=True")

        train_loader = DataLoader(
            train_ds, batch_size=128, shuffle=True, collate_fn=enhanced_collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=128, shuffle=False, collate_fn=enhanced_collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=128, shuffle=False, collate_fn=enhanced_collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

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
            epochs=100
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

        # Stage 3: Train PyTorch Tabular GATE
        print("\n  [Stage 3] Training PyTorch Tabular GATE...")
        ratio = np.sum(y_train==0) / (np.sum(y_train==1) + 1e-6)

        # Prepare data for PyTorch Tabular
        feature_cols = [f"feature_{i}" for i in range(X_train.shape[1])]

        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df['target'] = y_train

        val_df = pd.DataFrame(X_val, columns=feature_cols)
        val_df['target'] = y_val

        test_df = pd.DataFrame(X_test, columns=feature_cols)
        test_df['target'] = y_test

        # Configure GATE model
        data_config = DataConfig(
            target=['target'],
            continuous_cols=feature_cols,
            categorical_cols=[]
        )

        trainer_config = TrainerConfig(
            max_epochs=100,
            batch_size=256,
            early_stopping='valid_loss',
            early_stopping_patience=10,
            checkpoints='valid_loss',
            load_best=True,
            progress_bar='none',
            trainer_kwargs=dict(enable_model_summary=False)
        )

        optimizer_config = OptimizerConfig()

        model_config = GatedAdditiveTreeEnsembleConfig(
            task="classification",
            gflu_stages=6,
            gflu_dropout=0.0,
            tree_depth=5,
            num_trees=20,
            chain_trees=True,
            tree_dropout=0.1,
            share_head_weights=True
        )

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        tabular_model.fit(train=train_df, validation=val_df)

        # Evaluate hybrid model
        pred_df = tabular_model.predict(test_df)
        y_prob = pred_df['target_probability'].values
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

        # Baseline model with GATE
        print("\n  [Baseline] Training standard GATE...")
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

        # Prepare baseline data for PyTorch Tabular
        baseline_feature_cols = [f"feature_{i}" for i in range(X_tr_b.shape[1])]

        train_base_df = pd.DataFrame(X_tr_b.values, columns=baseline_feature_cols)
        train_base_df['target'] = y_tr_b.values

        test_base_df = pd.DataFrame(X_te_b.values, columns=baseline_feature_cols)
        test_base_df['target'] = y_te_b.values

        # Configure baseline GATE model
        baseline_data_config = DataConfig(
            target=['target'],
            continuous_cols=baseline_feature_cols,
            categorical_cols=[]
        )

        baseline_model = TabularModel(
            data_config=baseline_data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        baseline_model.fit(train=train_base_df)

        # Evaluate baseline model
        pred_base_df = baseline_model.predict(test_base_df)
        y_prob_b = pred_base_df['target_probability'].values
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

    ax1.set_title("Enhanced TBoostv2 with PyTorch Tabular GATE (New Data Format)")
    ax2.set_title("Baseline PyTorch Tabular GATE (New Data Format)")

    plt.tight_layout()
    plt.savefig("result/tboostv2_new_data_vs_baseline_gate.png", dpi=300)
    print("\nPlot saved to: result/tboostv2_new_data_vs_baseline_gate.png")

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
    #     print("\nUsage: python run_new_gate.py <path_to_joblib_file>")
    #     print("\nExample:")
    #     print("  python run_new_gate.py data/patients_new_format.joblib")
    #     sys.exit(1)

    data_filepath = NEW_DATA_PATH

    if not os.path.exists(data_filepath):
        print(f"\nError: File not found: {data_filepath}")
        sys.exit(1)

    main(data_filepath)
