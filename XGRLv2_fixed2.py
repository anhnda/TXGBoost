"""
REINFORCEMENT LEARNING V2 FIXED (v2): RNN Policy Network → XGBoost Judge

Critical fixes for cold start problem:
1. FIX ENTROPY BONUS (was backwards!)
2. Reduce latent_dim (28 → 16) to reduce noise
3. Add quick warm-up (5 epochs) to initialize policy
4. Start with stronger XGBoost to help learning
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import random
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

from xgboost import XGBClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

xseed = 42
seed_everything(xseed)

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData
from TimeEmbeddingVal import (
    get_all_temporal_features,
    extract_temporal_data,
    load_and_prepare_patients,
    split_patients_train_val,
)
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

# ==============================================================================
# Copy classes from XGRLv2_fixed.py
# ==============================================================================

class SimpleStaticEncoder:
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts = {f: 0 for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, 'values') and len(val) > 0:
                    val = list(val.values())[0]
                elif hasattr(val, 'values'):
                    val = 0.0
                val_str = str(val)
                try:
                    float(val)
                except ValueError:
                    if val_str not in self.mappings[f]:
                        self.mappings[f][val_str] = float(self.counts[f])
                        self.counts[f] += 1

    def transform(self, patient):
        vec = []
        for f in self.features:
            val = patient.measures.get(f, 0.0)
            if hasattr(val, 'values') and len(val) > 0:
                val = list(val.values())[0]
            elif hasattr(val, 'values'):
                val = 0.0
            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec

class HybridDataset(Dataset):
    def __init__(self, patients, feature_names, static_encoder, normalization_stats=None):
        self.data = []
        self.labels = []
        self.static_data = []
        self.feature_names = feature_names
        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue
            s_vec = static_encoder.transform(patient)
            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.data.append({'times': times, 'values': values, 'masks': masks})
            self.labels.append(1 if patient.akdPositive else 0)
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_values.append(v)

        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        for i in range(len(self.data)):
            norm_values = []
            for v_vec, m_vec in zip(self.data[i]['values'], self.data[i]['masks']):
                norm = [(v - self.mean)/self.std if m>0 else 0.0 for v, m in zip(v_vec, m_vec)]
                norm_values.append(norm)
            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.static_data[idx]

def hybrid_collate_fn(batch):
    data_list, label_list, static_list = zip(*batch)
    lengths = [len(d['times']) for d in data_list]
    max_len = max(lengths)
    feat_dim = data_list[0]['values'].shape[-1]
    batch_size = len(data_list)
    padded_times = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feat_dim)
    padded_masks = torch.zeros(batch_size, max_len, feat_dim)
    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_times[i, :l] = d['times']
        padded_values[i, :l] = d['values']
        padded_masks[i, :l] = d['masks']
    temporal_batch = {
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths)
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

class RNNPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)
        h = self.rnn_cell(times, values, masks, lengths)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std) * temperature
        policy_dist = dist.Normal(mean, std)
        if deterministic:
            z = mean
            log_prob = None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)
        return z, log_prob, mean, std

def extract_enriched_features_and_logprobs(policy_net, loader, deterministic=False, temperature=1.0):
    policy_net.eval() if deterministic else policy_net.train()
    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            z, log_prob, mean, std = policy_net(t_data, deterministic=deterministic, temperature=temperature)
            z_np = (mean if deterministic else z).detach().cpu().numpy()
            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()

            batch_last_vals = []
            batch_mean_vals = []
            batch_std_vals = []

            for i in range(len(vals)):
                patient_last = []
                patient_mean = []
                patient_std = []
                for f_idx in range(vals.shape[2]):
                    f_vals = vals[i, :, f_idx]
                    f_mask = masks[i, :, f_idx]
                    valid_idx = np.where(f_mask > 0)[0]
                    if len(valid_idx) > 0:
                        valid_vals = f_vals[valid_idx]
                        patient_last.append(valid_vals[-1])
                        patient_mean.append(np.mean(valid_vals))
                        patient_std.append(np.std(valid_vals) if len(valid_vals) > 1 else 0.0)
                    else:
                        patient_last.append(0.0)
                        patient_mean.append(0.0)
                        patient_std.append(0.0)
                batch_last_vals.append(patient_last)
                batch_mean_vals.append(patient_mean)
                batch_std_vals.append(patient_std)

            last_vals_arr = np.array(batch_last_vals)
            mean_vals_arr = np.array(batch_mean_vals)
            std_vals_arr = np.array(batch_std_vals)
            s_np = s_data.numpy()
            combined = np.hstack([s_np, last_vals_arr, mean_vals_arr, std_vals_arr, z_np])
            all_features.append(combined)
            all_labels.extend(labels.numpy())
            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None
    return features, labels, log_probs

# ==============================================================================
# NEW: Quick Warm-up
# ==============================================================================

def quick_warmup(policy_net, train_loader, val_loader, epochs=5):
    """Quick supervised warm-up to give policy a good start"""
    print(f"  [Quick Warmup] {epochs} epochs of supervised learning...")

    head = nn.Linear(policy_net.latent_dim + len(FIXED_FEATURES), 1).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(head.parameters()),
        lr=0.005  # Moderate LR
    )
    criterion = nn.BCELoss()

    best_aupr = 0
    best_state = None

    for epoch in range(epochs):
        policy_net.train()
        head.train()

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            z, _, _, _ = policy_net(t_data, deterministic=True)
            combined = torch.cat([z, s_data], dim=1)
            preds = torch.sigmoid(head(combined)).squeeze(-1)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Quick validation
        if (epoch + 1) % 2 == 0:
            policy_net.eval()
            head.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    z, _, _, _ = policy_net(t_data, deterministic=True)
                    combined = torch.cat([z, s_data], dim=1)
                    preds = torch.sigmoid(head(combined)).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_aupr = average_precision_score(all_labels, all_preds)
            print(f"    Warmup Epoch {epoch+1} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_aupr:
                best_aupr = val_aupr
                best_state = copy.deepcopy(policy_net.state_dict())

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    print(f"  [Quick Warmup] Done. Best AUPR: {best_aupr:.4f}")
    return policy_net

# ==============================================================================
# FIXED RL Training
# ==============================================================================

def train_policy_fixed_rl_v2(
    policy_net,
    train_loader,
    val_loader,
    xgb_params,
    epochs=100,
    update_xgb_every=5
):
    """
    CRITICAL FIXES:
    1. Entropy bonus fixed (was backwards!)
    2. Per-sample rewards only
    3. Good learning rate
    """

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0005)
    best_val_auc = 0
    best_state = None
    patience = 20
    patience_counter = 0

    print("  [FIXED RL v2] Training with ALL fixes applied...")
    xgb_model = None

    for epoch in range(epochs):
        temperature = max(0.5, 1.0 - epoch / (epochs * 2.0))
        policy_net.train()

        # Sample features
        X_train, y_train, log_probs_train = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        # Train XGBoost
        if epoch % update_xgb_every == 0 or xgb_model is None:
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train)

        # Per-sample rewards
        y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)
        y_train_pred = (y_train_proba > 0.5).astype(int)
        rewards_binary = (y_train_pred == y_train).astype(np.float32)
        rewards_combined = 0.5 * rewards_binary + 0.5 * rewards_smooth

        # Normalize
        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Policy gradient
        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        # CRITICAL FIX: Entropy bonus with correct sign!
        # log_probs are negative, so we ADD them to encourage exploration
        entropy_regularization = 0.01 * log_probs_train.mean()  # Negative value
        total_loss = policy_loss + entropy_regularization  # Adding negative = subtracting positive = more exploration

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_enriched_features_and_logprobs(
                    policy_net, val_loader, deterministic=True
                )
                X_train_det, y_train_det, _ = extract_enriched_features_and_logprobs(
                    policy_net, train_loader, deterministic=True
                )

                xgb_val_model = XGBClassifier(**xgb_params)
                xgb_val_model.fit(X_train_det, y_train_det)
                y_val_proba_det = xgb_val_model.predict_proba(X_val_det)[:, 1]

                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | Reward: {rewards_combined.mean():.4f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f}")

                if val_aupr_det > best_val_auc:
                    best_val_auc = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("="*80)
    print("RL POLICY V2 FIXED (v2) with XGBoost")
    print("Fixes: Entropy, Warm-up, Smaller latent_dim")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    metrics_rl = {k: [] for k in ['auc', 'auc_pr']}
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # FIX: Smaller latent_dim to reduce noise
        latent_dim = 16  # Was 28, now 16
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats),
            hidden_dim=16,  # Was 20, now 16
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        ratio = np.sum(train_ds.labels == 0) / (np.sum(train_ds.labels == 1) + 1e-6)
        xgb_params = {
            'n_estimators': 400,  # Strong XGBoost
            'max_depth': 5,
            'learning_rate': 0.05,
            'scale_pos_weight': ratio,
            'eval_metric': 'auc',
            'random_state': 42,
            'verbosity': 0
        }

        # Quick warm-up
        policy_net = quick_warmup(policy_net, train_loader, val_loader, epochs=5)

        # RL Training
        policy_net = train_policy_fixed_rl_v2(
            policy_net,
            train_loader,
            val_loader,
            xgb_params,
            epochs=100,
            update_xgb_every=5
        )

        # Final Evaluation
        print("\n  [Final Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_enriched_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_enriched_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        final_xgb = XGBClassifier(**xgb_params)
        final_xgb.fit(X_train_final, y_train_final)

        y_test_proba = final_xgb.predict_proba(X_test_final)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl['auc'].append(fold_auc)
        metrics_rl['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  RL Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

    ax1.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("RL Policy V2 Fixed (v2) + XGBoost", fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("result/xgrl_v2_fixed2.png", dpi=300)
    print("\nPlot saved to result/xgrl_v2_fixed2.png")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    rl_mean, rl_std = np.mean(metrics_rl['auc']), np.std(metrics_rl['auc'])
    aupr_mean, aupr_std = np.mean(metrics_rl['auc_pr']), np.std(metrics_rl['auc_pr'])
    print(f"AUC             | RL: {rl_mean:.4f} ± {rl_std:.4f}")
    print(f"AUC-PR          | RL: {aupr_mean:.4f} ± {aupr_std:.4f}")

if __name__ == "__main__":
    main()
