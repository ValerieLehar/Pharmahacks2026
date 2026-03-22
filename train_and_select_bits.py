import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import shap
import json

# ── CONFIG ───────────────────────────────────────────────────────────────
CSV_PATH = "morgan_fingerprints.csv"
TOP_SAVE = 50
META_COLS = ["SMILES", "Target", "amino_acid_sequence", "Affinity", "Molecule"]

# ── 1. Load training data ────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
bit_cols = [c for c in df.columns if c not in META_COLS]

X = df[bit_cols].values.astype(np.float32)
y = df["Affinity"].values.astype(np.float32)

# ── 2. Train model ───────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)

# ── 3. Compute SHAP importance ───────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_importance = np.abs(shap_values).mean(axis=0)

df_shap = pd.DataFrame({
    "feature": bit_cols,
    "shap_importance": shap_importance,
}).sort_values("shap_importance", ascending=False)

# ── 4. Save top bits to JSON ─────────────────────────────────────────────
top_bits = df_shap.head(TOP_SAVE)["feature"].tolist()

with open("selected_bits.json", "w") as f:
    json.dump({"top_bits": top_bits}, f, indent=2)

print(f"Saved top {TOP_SAVE} bits → selected_bits.json")