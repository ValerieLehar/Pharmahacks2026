import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 1. Load both CSVs ─────────────────────────────────────────────────────────
rdkit   = pd.read_csv("filtered_RDKit_feat.csv")
morgan  = pd.read_csv("morgan_fingerprints_top50_shap.csv")

# ── 2. Scale RDKit features (skip first 2 columns) ───────────────────────────
meta_rdkit    = rdkit.iloc[:, :2]          # first 2 cols kept as-is
features_rdkit = rdkit.iloc[:, 2:]         # columns to scale

rdkit_processed = pd.concat([meta_rdkit.reset_index(drop=True),
                              features_rdkit.reset_index(drop=True)], axis=1)

# ── 3. Extract Morgan bit columns (col index 5 onwards, i.e. 6th col+) ───────
morgan_bits = morgan.iloc[:, 5:].reset_index(drop=True)

# ── 4. Concatenate and save ───────────────────────────────────────────────────
combined = pd.concat([rdkit_processed, morgan_bits], axis=1)

combined.to_csv("combined_features.csv", index=False)

print(f"Final CSV dimensions: {combined.shape[0]} rows × {combined.shape[1]} columns")
print(f"  RDKit features : {features_rdkit.shape[1]}")
print(f"  Morgan bit features   : {morgan_bits.shape[1]}")
print(f"  Meta columns (RDKit)  : {meta_rdkit.shape[1]}")

