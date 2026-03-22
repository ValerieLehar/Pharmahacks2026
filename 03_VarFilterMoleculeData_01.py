import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor


# 1. Load CSV with RDKit features already appended
df = pd.read_csv("molecular_descriptors_with_affinity.csv")
#df = df._append(pd.read_csv("molecular_descriptors.csv"))

# Core columns from the challenge
core_cols = ["SMILES", "Affinity"]

missing = [c for c in core_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# 2. Define X (features) and y (target)
#    Use all non-core columns as features (these should be your RDKit descriptors etc.)
feature_cols = [c for c in df.columns if c not in core_cols]
X_full = df[feature_cols]
y = df["Affinity"].astype(float)

# XGBoost can handle NaNs, but VarianceThreshold cannot; fill or drop as you prefer.
# Here we fill NaNs with column medians (simple, fast).
X_full = X_full.copy()
for col in feature_cols:
    if X_full[col].isna().any():
        X_full[col].fillna(X_full[col].median(), inplace=True)


# 2.5 Fix crash by filtering out infinities before variance filtering
# Replace inf / -inf with NaN, then fill with median
mask_inf = np.isinf(X_full).any()
print("Columns with inf:", X_full.columns[mask_inf].tolist())

import numpy as np
import pandas as pd

def clip_infinities(df):
    df = df.copy()
    for col in df.columns:
        col_values = df[col].values

        # Identify finite values
        finite_mask = np.isfinite(col_values)
        if not finite_mask.any():
            # Column is entirely inf/NaN — drop or handle separately
            print(f"Column '{col}' has no finite values. Consider dropping it.")
            continue

        finite_min = col_values[finite_mask].min()
        finite_max = col_values[finite_mask].max()

        # Replace +inf with finite_max, -inf with finite_min
        df[col] = np.where(col_values == np.inf, finite_max, col_values)
        df[col] = np.where(col_values == -np.inf, finite_min, df[col])

    return df



X_full = clip_infinities(X_full)
for col in feature_cols:
    if X_full[col].isna().any():
        X_full[col] = X_full[col].fillna(X_full[col].median())


# 3. Remove low-variance features (variance < 0.001)
#    For binary bits in [0,1], this roughly corresponds to features active in <~0.4% of compounds.
var_thresh = 1e-3
selector = VarianceThreshold(threshold=var_thresh)
X_reduced = selector.fit_transform(X_full.values)

# Map back to column names
kept_mask = selector.get_support()
kept_feature_names = [name for name, keep in zip(feature_cols, kept_mask) if keep]

print(f"Original feature count: {len(feature_cols)}")
print(f"Kept {len(kept_feature_names)} features after variance filtering (threshold={var_thresh}).")

# 4. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_reduced,
    y.values,
    test_size=0.2,
    random_state=42,
)

# 5. Define and fit XGBRegressor
#    This is a reasonable starting configuration; you can tune later.
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    #eval_metric="rmse",
    verbose=False,  # set True if you want per-iteration logs
)

# 6. Evaluate on validation set
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation MAE: {mae:.4f}")
print(f"Validation R^2: {r2:.4f}")

# 7. (Optional) Attach kept feature names back to the model or save them
#    so you can apply the same filtering at inference time.
kept_features_df = pd.DataFrame({"feature": kept_feature_names})
kept_features_df.to_csv("kept_features_after_variance_threshold.csv", index=False)
print("Saved kept feature names to kept_features_after_variance_threshold.csv")