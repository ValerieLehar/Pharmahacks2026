import pandas as pd
import numpy as np

# === CONFIG ===
input_csv = "MOL-with_DES.csv"          # change to your actual input file
output_csv = "output_no_inf.csv" # desired output file name


# Read CSV; assume:
# col 0 = SMILES (string), col 1 = Affinity (numeric), rest = numeric RDKit features
df = pd.read_csv(input_csv)

# Identify numeric columns (Affinity + RDKit features).
# SMILES should be non-numeric and excluded automatically.
numeric_cols = df.select_dtypes(include=["number"]).columns
print("Numeric columns:", list(numeric_cols))

# Find columns that contain +/- infinity
inf_cols = []
for col in numeric_cols:
    col_vals = df[col].to_numpy()
    if np.isinf(col_vals).any():
        inf_cols.append(col)
        pos_inf = np.isposinf(col_vals).sum()
        neg_inf = np.isneginf(col_vals).sum()
        print(f"Column '{col}' has {pos_inf} +inf and {neg_inf} -inf values")

if not inf_cols:
    print("No infinities found in numeric columns.")

# For each column with infinities, compute finite min/max and replace
for col in inf_cols:
    col_vals = df[col].to_numpy()
    finite_mask = np.isfinite(col_vals)

    if not finite_mask.any():
        # Edge case: column is all inf/NaN; you can decide what to do here
        print(f"Warning: column '{col}' has no finite values; leaving as-is")
        continue

    finite_vals = col_vals[finite_mask]
    finite_min = finite_vals.min()
    finite_max = finite_vals.max()
    print(f"Column '{col}': finite min={finite_min}, finite max={finite_max}")

    # Replace +inf with finite_max, -inf with finite_min
    col_vals[np.isposinf(col_vals)] = finite_max
    col_vals[np.isneginf(col_vals)] = finite_min
    df[col] = col_vals

# Save cleaned dataframe to new CSV
df.to_csv(output_csv, index=False)
print(f"Cleaned CSV saved to: {output_csv}")