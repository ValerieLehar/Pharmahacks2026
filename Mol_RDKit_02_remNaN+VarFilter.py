import pandas as pd
import numpy as np


def filter_low_variance_features(
    input_csv: str,
    output_csv: str,
    variance_threshold: float = 0.001,
) -> None:
    """
    Read input_csv, keep first two columns (SMILES, affinity),
    replace NaNs in feature columns with 0, drop feature columns
    whose variance is < variance_threshold, and save to output_csv.
    """

    # Load data
    df = pd.read_csv(input_csv)

    # First two columns: SMILES, affinity
    meta = df.iloc[:, :2]
    features = df.iloc[:, 2:]

    # Ensure we only work on numeric feature columns (non-numeric kept as-is later if you prefer)
    # If all feature columns are numeric already, this is effectively a no-op cast.
    features = features.apply(pd.to_numeric, errors="coerce")

    # Replace NaNs with 0 before variance computation (and for downstream PCA)
    features = features.fillna(0)

    # Compute per-feature variance (population variance, like sklearn.VarianceThreshold)[web:19][web:24]
    variances = features.var(axis=0, ddof=0)

    # Boolean mask of features to keep (variance >= threshold)
    kept_mask = variances >= variance_threshold

    # Apply mask
    filtered_features = features.loc[:, kept_mask]

    # Concatenate SMILES/affinity back
    out_df = pd.concat([meta, filtered_features], axis=1)

    # Save result
    out_df.to_csv(output_csv, index=False)

    # Simple summary
    n_before = features.shape[1]
    n_after = filtered_features.shape[1]
    print(
        f"Kept {n_after} / {n_before} feature columns "
        f"with variance ≥ {variance_threshold}."
    )


def main():
    # EDIT THESE PATHS / THRESHOLD AND THEN JUST RUN THIS FILE
    input_csv = "output_no_inf.csv"        # path to your original CSV
    output_csv = "filtered.csv"    # path for the filtered CSV
    variance_threshold = 0.001     # 0.001 ≈ 0.1% variance if features are scaled ~[0,1]

    filter_low_variance_features(
        input_csv=input_csv,
        output_csv=output_csv,
        variance_threshold=variance_threshold,
    )


if __name__ == "__main__":
    main()