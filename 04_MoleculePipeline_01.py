import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from multiprocessing import Pool, cpu_count

from sklearn.feature_selection import VarianceThreshold


# --------------------------
# RDKit descriptor calculation
# --------------------------

# Precompute descriptor names and calculator once
DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


def _calc_descriptors_for_smiles(smiles: str):
    """Return dict(name -> value) for one SMILES; empty dict if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    values = CALCULATOR.CalcDescriptors(mol)
    return dict(zip(DESCRIPTOR_NAMES, values))


def compute_rdkit_descriptors(smiles_series: pd.Series, n_procs: int | None = None) -> pd.DataFrame:
    """
    Compute RDKit 2D descriptors for each SMILES in a Series.
    Returns a DataFrame aligned with the input index.
    """
    smiles_list = smiles_series.tolist()
    if n_procs is None:
        n_procs = max(1, cpu_count() - 1)

    with Pool(n_procs) as pool:
        results = pool.map(_calc_descriptors_for_smiles, smiles_list)

    desc_df = pd.DataFrame(results, index=smiles_series.index)
    return desc_df


# --------------------------
# Cleaning helpers
# --------------------------

def replace_infinities_with_finite_extrema(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric column:
    - Replace +inf with finite max of that column
    - Replace -inf with finite min of that column
    Columns with no finite values are left as‑is (and could later be dropped).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        col_vals = df[col].to_numpy()
        # Identify finite values
        finite_mask = np.isfinite(col_vals)
        if not finite_mask.any():
            # Column is entirely inf/NaN — leave as is or drop later
            print(f"Warning: column '{col}' has no finite values; leaving as‑is.")
            continue

        finite_vals = col_vals[finite_mask]
        finite_min = finite_vals.min()
        finite_max = finite_vals.max()

        # Replace infinities
        col_vals[np.isposinf(col_vals)] = finite_max
        col_vals[np.isneginf(col_vals)] = finite_min
        df[col] = col_vals

    return df


def fill_nans_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill all NaN values with zero in numeric columns.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    return df


def variance_filter(
    df: pd.DataFrame,
    core_cols: list[str] = ("SMILES", "Affinity"),
    var_threshold: float = 1e-3,
) -> pd.DataFrame:
    """
    Apply VarianceThreshold to non-core numeric columns and drop low-variance ones.
    Keeps all core columns unchanged.
    """
    df = df.copy()

    # Ensure core columns exist but do not crash if they don't
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing expected core columns: {missing_core}")

    # Features = all numeric columns except core columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in core_cols]

    if not feature_cols:
        print("No feature columns (beyond core) found; skipping variance filter.")
        return df

    X_full = df[feature_cols].to_numpy()

    selector = VarianceThreshold(threshold=var_threshold)
    X_reduced = selector.fit_transform(X_full)

    kept_mask = selector.get_support()
    kept_feature_names = [name for name, keep in zip(feature_cols, kept_mask) if keep]

    print(f"Original feature count (excluding core): {len(feature_cols)}")
    print(f"Kept {len(kept_feature_names)} features after variance filtering (threshold={var_threshold}).")

    # Build final DataFrame: core columns + kept features
    kept_df = pd.DataFrame(X_reduced, columns=kept_feature_names, index=df.index)
    out_df = pd.concat([df[list(core_cols)], kept_df], axis=1)

    return out_df


# --------------------------
# Main pipeline
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute RDKit descriptors from an input CSV, "
            "clean infinities/NaNs, then drop low‑variance features."
        )
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file with at least 'SMILES' and 'Affinity' columns.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file for cleaned, variance‑filtered data.",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=1e-3,
        help="Variance threshold for dropping low‑variance features (default: 1e-3 ≈ 0.1%% for [0,1] bits).",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=None,
        help="Number of processes for RDKit descriptor calculation (default: cpu_count() - 1).",
    )

    args = parser.parse_args()

    # 1. Load input
    df = pd.read_csv(args.input)

    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")
    if "Affinity" not in df.columns:
        raise ValueError("Input CSV must contain an 'Affinity' column.")

    # 2. Compute RDKit descriptors
    print("Computing RDKit descriptors...")
    desc_df = compute_rdkit_descriptors(df["SMILES"], n_procs=args.n_procs)

    # 3. Merge descriptors back to original data
    print("Merging descriptors with original data...")
    full_df = pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

    # 4. Replace +/-inf with column finite min/max
    print("Replacing infinities with finite min/max...")
    full_df = replace_infinities_with_finite_extrema(full_df)

    # 5. Replace remaining NaNs with zero (numeric columns)
    print("Filling NaN with zero in numeric columns...")
    full_df = fill_nans_with_zero(full_df)

    # 6. Variance filter (drop low-variance non-core features)
    print("Applying variance filter...")
    final_df = variance_filter(
        full_df,
        core_cols=["SMILES", "Affinity"],
        var_threshold=args.var_threshold,
    )

    # 7. Save result
    final_df.to_csv(args.output, index=False)
    print(f"Done. Saved cleaned, filtered data to: {args.output}")


if __name__ == "__main__":
    main()