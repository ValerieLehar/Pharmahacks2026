
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
MOLECULE_CSV  = "Molecule_combined_features2.csv"
PROTEIN_CSV   = "protein_embeddings_full.csv"   



LATENT_DIM    = 128
DROPOUT_RATE  = 0.3
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
MAX_EPOCHS    = 200
PATIENCE      = 15          # early stopping patience (epochs without val improvement)
VAL_SPLIT     = 0.10
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")





# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
VAL_SPLIT = 0.15   # 15% validation

mol_df  = pd.read_csv(MOLECULE_CSV)
prot_df = pd.read_csv(PROTEIN_CSV)

# Replace NaNs (safe default)
mol_df  = mol_df.fillna(0)
prot_df = prot_df.fillna(0)

# Quick NaN diagnostic
print("\n=== Checking for NaNs in raw CSVs ===")
print("Mol_df NaNs:", mol_df.isna().sum().sum())
print("Prot_df NaNs:", prot_df.isna().sum().sum())

assert len(mol_df) == len(prot_df), "Molecule and protein CSVs must have same number of rows."

# Column layout: [SMILES | pIC50 | feature_0 ... feature_N]
y      = mol_df.iloc[:, 1].values.astype(np.float32)
X_mol  = mol_df.iloc[:,  2:].values.astype(np.float32)
X_prot = prot_df.iloc[:, 2:].values.astype(np.float32)

N = len(y)
MOL_DIM  = X_mol.shape[1]
PROT_DIM = X_prot.shape[1]

print(f"Samples: {N} | Mol features: {MOL_DIM} | Prot features: {PROT_DIM}")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. TRAIN / VALIDATION SPLIT (15% validation)
# ═══════════════════════════════════════════════════════════════════════════════

train_idx, val_idx = train_test_split(
    np.arange(N),
    test_size=VAL_SPLIT,
    random_state=SEED,
    shuffle=True
)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. DETECT BINARY VS CONTINUOUS COLUMNS
#     3.1 Concordia index function
# ═══════════════════════════════════════════════════════════════════════════════

def detect_binary_columns(X):
    """
    Returns indices of columns that contain ONLY 0/1 values.
    """
    binary_cols = []
    for col in range(X.shape[1]):
        unique_vals = np.unique(X[:, col])
        if np.all(np.isin(unique_vals, [0, 1])):
            binary_cols.append(col)
    return np.array(binary_cols, dtype=int)

mol_binary_cols  = detect_binary_columns(X_mol)
prot_binary_cols = detect_binary_columns(X_prot)

mol_cont_cols  = np.setdiff1d(np.arange(MOL_DIM), mol_binary_cols)
prot_cont_cols = np.setdiff1d(np.arange(PROT_DIM), prot_binary_cols)

print(f"Molecule: {len(mol_binary_cols)} binary, {len(mol_cont_cols)} continuous")
print(f"Protein : {len(prot_binary_cols)} binary, {len(prot_cont_cols)} continuous")

def concordance_index(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff_true = y_true[:, None] - y_true[None, :]
    diff_pred = y_pred[:, None] - y_pred[None, :]
    mask = diff_true != 0
    concordant = np.sum((diff_pred[mask] * diff_true[mask] > 0)) + \
                 0.5 * np.sum(diff_pred[mask] == 0)
    return concordant / mask.sum() if mask.sum() > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SCALE ONLY CONTINUOUS COLUMNS (NO LEAKAGE)
# ═══════════════════════════════════════════════════════════════════════════════

mol_scaler  = StandardScaler().fit(X_mol[train_idx][:, mol_cont_cols])
prot_scaler = StandardScaler().fit(X_prot[train_idx][:, prot_cont_cols])

# Apply scaling
X_mol_sc  = X_mol.copy()
X_prot_sc = X_prot.copy()

X_mol_sc[:, mol_cont_cols]  = mol_scaler.transform(X_mol[:, mol_cont_cols])
X_prot_sc[:, prot_cont_cols] = prot_scaler.transform(X_prot[:, prot_cont_cols])

print("\nScaling complete. Binary columns untouched. Continuous columns standardized.")


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  DATASET & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════
class DTIDataset(Dataset):
    def __init__(self, X_mol, X_prot, y, idx):
        self.X_mol  = torch.tensor(X_mol[idx],  dtype=torch.float32)
        self.X_prot = torch.tensor(X_prot[idx], dtype=torch.float32)
        self.y      = torch.tensor(y[idx],      dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_mol[i], self.X_prot[i], self.y[i]

train_ds = DTIDataset(X_mol_sc, X_prot_sc, y, train_idx)
val_ds   = DTIDataset(X_mol_sc, X_prot_sc, y, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class EncoderTower(nn.Module):
    """
    Single encoder tower.
    Architecture:  input_dim → h1 → h2 → latent_dim
                               ↑── 3 total linear layers (3 "hidden" projections)
    Each hidden layer:  Linear → BatchNorm → ReLU → Dropout
    Latent layer:       Linear → BatchNorm → ReLU  (no dropout on the embedding)
    """
    def __init__(self, input_dim: int, h1: int, h2: int,
                 latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            # ── Hidden layer 1 ──────────────────────────────────────────────
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            # ── Hidden layer 2 ──────────────────────────────────────────────
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # ── Latent projection (hidden layer 3) ──────────────────────────
            nn.Linear(h2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),          # no dropout — keep the embedding clean
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)      # shape: (B, latent_dim)


class FusionMLP(nn.Module):
    """
    Deep fusion head.
    Architecture:  [z_mol ‖ z_prot](256) → 128 → 64 → 32 → 8 → 1
    Each hidden layer: Linear → ReLU → Dropout
    Output layer: Linear only (raw pIC50 regression)
    """
    def __init__(self, latent_dim: int, dropout: float):
        super().__init__()
        D = latent_dim * 2      # 256 after concatenation
        self.net = nn.Sequential(
            nn.Linear(D,   128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,   32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32,    8), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(8,     1),            # ← scalar pIC50 output
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)      # shape: (B, 1)


class TwoTowerDTI(nn.Module):
    """
    Full two-tower model.

                 X_mol ──► MolTower ──► z_mol (128)  ─┐
                                                        cat(256) ──► FusionMLP ──► pIC50
                X_prot ──► ProtTower ──► z_prot(128) ─┘
    """
    def __init__(self, mol_dim: int, prot_dim: int,
                 latent_dim: int, dropout: float):
        super().__init__()
        # Molecule tower: 253 → 512 → 256 → 128
        self.mol_tower  = EncoderTower(mol_dim,  512,  256, latent_dim, dropout)
        # Protein tower:  1022 → 512 → 256 → 128
        self.prot_tower = EncoderTower(prot_dim, 512,  256, latent_dim, dropout)
        self.fusion     = FusionMLP(latent_dim, dropout)

    def forward(self, x_mol: torch.Tensor,
                x_prot: torch.Tensor) -> torch.Tensor:
        z_mol  = self.mol_tower(x_mol)              # (B, 128)
        z_prot = self.prot_tower(x_prot)            # (B, 128)
        z      = torch.cat([z_mol, z_prot], dim=1)  # (B, 256)
        return self.fusion(z)                       # (B, 1)

    def encode(self, x_mol: torch.Tensor,
               x_prot: torch.Tensor):
        """Return latent embeddings (useful for probing/visualisation)."""
        with torch.no_grad():
            return self.mol_tower(x_mol), self.prot_tower(x_prot)


model    = TwoTowerDTI(MOL_DIM, PROT_DIM, LATENT_DIM, DROPOUT_RATE).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel architecture:\n{model}")
print(f"\nTrainable parameters: {n_params:,}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Halve LR whenever val loss plateaus for 5 consecutive epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
)


def run_epoch(model, loader, optimizer, criterion, device, training: bool):
    model.train() if training else model.eval()
    total_loss, preds_all, targets_all = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x_mol, x_prot, y_batch in loader:
            x_mol, x_prot, y_batch = (
                x_mol.to(device), x_prot.to(device), y_batch.to(device)
            )
            preds = model(x_mol, x_prot)
            loss  = criterion(preds, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(y_batch.cpu().numpy())

    preds_all   = np.concatenate(preds_all).flatten()
    targets_all = np.concatenate(targets_all).flatten()

    avg_mse = total_loss / len(loader.dataset)
    rmse    = np.sqrt(avg_mse)
    pear, _ = pearsonr(targets_all, preds_all)
    spear, _ = spearmanr(targets_all, preds_all)
    ci      = concordance_index(targets_all, preds_all)

    return avg_mse, rmse, pear, spear, ci, \
           r2_score(targets_all, preds_all), \
           mean_absolute_error(targets_all, preds_all)


# ── Main loop ─────────────────────────────────────────────────────────────────
best_val_loss, patience_count = float("inf"), 0
history = {
    "train_mse": [], "val_mse": [],
    "train_rmse": [], "val_rmse": [],
    "train_pear": [], "val_pear": [],
    "train_spear": [], "val_spear": [],
    "train_ci": [], "val_ci": [],
    "val_r2": [], "val_mae": []
}

print(f"\n{'Epoch':>6} | {'Train MSE':>10} | {'Val MSE':>10} | {'Val R²':>8} | {'Val MAE':>8}")
print("─" * 57)

for epoch in range(1, MAX_EPOCHS + 1):
    tr_mse, tr_rmse, tr_pear, tr_spear, tr_ci, _, _ = run_epoch(
    model, train_loader, optimizer, criterion, DEVICE, training=True
    )

    va_mse, va_rmse, va_pear, va_spear, va_ci, va_r2, va_mae = run_epoch(
    model, val_loader, optimizer, criterion, DEVICE, training=False
    )

    scheduler.step(va_mse)
    history["train_rmse"].append(tr_rmse)
    history["val_rmse"].append(va_rmse)

    history["train_pear"].append(tr_pear)
    history["val_pear"].append(va_pear)

    history["train_spear"].append(tr_spear)
    history["val_spear"].append(va_spear)

    history["train_ci"].append(tr_ci)
    history["val_ci"].append(va_ci)

    history["train_mse"].append(tr_mse)
    history["val_mse"].append(va_mse)
    
    history["val_r2"].append(va_r2)
    history["val_mae"].append(va_mae)

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6} | {tr_mse:>10.4f} | {va_mse:>10.4f} | {va_r2:>8.4f} | {va_mae:>8.4f}")

    # ── Early stopping ────────────────────────────────────────────────────────
    if va_mse < best_val_loss:
        best_val_loss  = va_mse
        patience_count = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

# ── Load best checkpoint & report ────────────────────────────────────────────
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
best_mse, best_rmse, best_pear, best_spear, best_ci, best_r2, best_mae = run_epoch(
    model, val_loader, optimizer, criterion, DEVICE, training=False
)

print(f"{'═'*42}")
print(f"  Best checkpoint — Validation Metrics")
print(f"  MSE      : {best_mse:.4f}")
print(f"  RMSE     : {best_rmse:.4f}")
print(f"  MAE      : {best_mae:.4f}")
print(f"  R²       : {best_r2:.4f}")
print(f"  Pearson  : {best_pear:.4f}")
print(f"  Spearman : {best_spear:.4f}")
print(f"  CI       : {best_ci:.4f}")
print(f"{'═'*42}")

# ---------------------------------------------------------
# INSERT STEP 5 (PLOTS) HERE
# ---------------------------------------------------------

import matplotlib.pyplot as plt

def plot_metric(history, key_train, key_val, title, ylabel):
    plt.figure(figsize=(7,5))
    plt.plot(history[key_train], label="Train")
    plt.plot(history[key_val], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_metric(history, "train_mse",   "val_mse",   "MSE over epochs",   "MSE")
plot_metric(history, "train_rmse",  "val_rmse",  "RMSE over epochs",  "RMSE")
plot_metric(history, "train_pear",  "val_pear",  "Pearson over epochs", "Pearson r")
plot_metric(history, "train_spear", "val_spear", "Spearman over epochs", "Spearman ρ")
plot_metric(history, "train_ci",    "val_ci",    "Concordance Index", "CI")


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  INFERENCE UTILITY  (for held-out test sets)
# ═══════════════════════════════════════════════════════════════════════════════
def predict(mol_df_test: pd.DataFrame,
            prot_df_test: pd.DataFrame) -> np.ndarray:
    """
    Takes two raw test DataFrames (same column layout as training CSVs),
    applies the fitted scalers, and returns pIC50 predictions as a 1-D array.
    """
    X_mol_t = mol_df_test.iloc[:, 2:].values.astype(np.float32)
    X_prot_t = prot_df_test.iloc[:, 2:].values.astype(np.float32)
    X_mol_t[:, mol_cont_cols]   = mol_scaler.transform(X_mol_t[:, mol_cont_cols])
    X_prot_t[:, prot_cont_cols] = prot_scaler.transform(X_prot_t[:, prot_cont_cols])

    ds     = DTIDataset(X_mol_t, X_prot_t,
                        np.zeros(len(mol_df_test), dtype=np.float32),
                        np.arange(len(mol_df_test)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for x_mol, x_prot, _ in loader:
            p = model(x_mol.to(DEVICE), x_prot.to(DEVICE))
            preds.append(p.cpu().numpy())
    return np.concatenate(preds).flatten()
