import pandas as pd
import json
import glob

META_COLS = ["SMILES", "Target", "amino_acid_sequence", "Affinity", "Molecule"]

# ── Load selected bits from training ─────────────────────────────────────
with open("selected_bits.json", "r") as f:
    top_bits = json.load(f)["top_bits"]

keep_cols = META_COLS + top_bits

# ── Process all test CSVs in a folder ────────────────────────────────────
for path in glob.glob("test_data/*.csv"):
    df = pd.read_csv(path)

    # Keep only columns that exist in the test file
    cols_present = [c for c in keep_cols if c in df.columns]
    df_trimmed = df[cols_present]

    out_path = path.replace(".csv", "_filtered.csv")
    df_trimmed.to_csv(out_path, index=False)

    print(f"Saved filtered file → {out_path}")