import os
import math
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, EState
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def build_descriptor_names():
    """
    Build the descriptor name list:
      - All explicit 2D/global descriptors from your list
      - All PEOE_VSA*, SlogP_VSA*, EState_VSA*, VSA_EState*
      - All fragment descriptors fr_*
    """
    all_desc_names = {name for name, _ in Descriptors._descList}

    explicit = {
        # Lipophilicity / size-polarizability
        "MolLogP", "MolMR", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",

        # H-bond / heteroatom / counts
        "NumHDonors", "NumHAcceptors", "NHOHCount", "NOCount",
        "NumHeteroatoms", "fr_halogen",

        # PSA / polarity
        "TPSA",

        # Mass & composition
        "ExactMolWt", "HeavyAtomCount", "NumValenceElectrons",
        "NumRadicalElectrons",

        # Drug-likeness / composite
        "qed", "RingCount", "FractionCSP3",
        "Phi", "SPS",  # present in RDKit descriptor set in recent versions

        # Topology / complexity
        "BertzCT", "BalabanJ", "Ipc", "AvgIpc",

        # Chi/Kier connectivity indices
        "Chi0", "Chi0n", "Chi0v",
        "Chi1", "Chi1n", "Chi1v",
        "Chi2n", "Chi2v",
        "Chi3n", "Chi3v",
        "Chi4n", "Chi4v",

        # Kappa / shape indices
        "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",

        # BCUT families
        "BCUT2D_MWHI", "BCUT2D_MWLOW",
        "BCUT2D_MRHI", "BCUT2D_MRLOW",
        "BCUT2D_CHGHI", "BCUT2D_CHGLO",

        # Flexibility / rings / scaffolds
        "NumRotatableBonds",
        "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
        "NumAliphaticRings",
        "NumAromaticCarbocycles", "NumAromaticHeterocycles",
        "NumAromaticRings",
        "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
        "NumSaturatedRings",
        "NumHeterocycles",
        "fr_bicyclic", "fr_benzene",
        "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters",
        "NumBridgeheadAtoms", "NumSpiroAtoms",

        # Charge/electronic distribution
        "MaxPartialCharge", "MinPartialCharge",
        "MaxAbsPartialCharge", "MinAbsPartialCharge",
        "MaxEStateIndex", "MinEStateIndex",
        "MaxAbsEStateIndex", "MinAbsEStateIndex",

        # Fingerprint density
        "FpDensityMorgan2",
    }

    # Families of surface-area descriptors and fragments
    descriptor_names = set()

    for name in all_desc_names:
        if (
            name in explicit
            or name.startswith("PEOE_VSA")
            or name.startswith("SlogP_VSA")
            or name.startswith("EState_VSA")
            or name.startswith("VSA_EState")
            or name.startswith("fr_")  # all fragment descriptors
        ):
            descriptor_names.add(name)

    # Keep only those that really exist (guards against RDKit version differences)
    descriptor_names = sorted(n for n in descriptor_names if n in all_desc_names)

    return descriptor_names


def compute_atom_level_aggregates(mol):
    """
    Compute mean and std for per-atom Gasteiger charges and EState indices.
    Returns a dict with keys:
      Charge_mean, Charge_std, EState_mean, EState_std
    """
    agg = {
        "Charge_mean": math.nan,
        "Charge_std": math.nan,
        "EState_mean": math.nan,
        "EState_std": math.nan,
    }

    if mol is None:
        return agg

    # Gasteiger charges
    try:
        ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                val = atom.GetProp("_GasteigerCharge")
                if val not in ("nan", "inf", "-inf"):
                    charges.append(float(val))
        if charges:
            charges = np.array(charges, dtype=float)
            agg["Charge_mean"] = float(charges.mean())
            agg["Charge_std"] = float(charges.std(ddof=0))
    except Exception:
        pass  # leave as NaN

    # EState indices
    try:
        est = EState.EStateIndices(mol)
        if est:
            est_arr = np.array(est, dtype=float)
            agg["EState_mean"] = float(est_arr.mean())
            agg["EState_std"] = float(est_arr.std(ddof=0))
    except Exception:
        pass

    return agg


def main(input_csv, output_csv="mfi_rdkit_descriptors.csv"):
    # 1. Load CSV
    df = pd.read_csv(input_csv)

    # Sanity check on columns
    expected_cols = ["SMILES", "Target", "amino_acid_sequence", "Affinity"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    # 2. Prepare descriptor calculator
    descriptor_names = build_descriptor_names()
    print(f"Using {len(descriptor_names)} RDKit descriptors.")
    calc = MolecularDescriptorCalculator(descriptor_names)

    # 3. Iterate over molecules and compute descriptors
    desc_rows = []

    for idx, row in df.iterrows():
        smi = row["SMILES"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None

        if mol is None:
            # Fill with NaNs for all descriptors when parsing fails
            desc_dict = {name: math.nan for name in descriptor_names}
            agg = {
                "Charge_mean": math.nan,
                "Charge_std": math.nan,
                "EState_mean": math.nan,
                "EState_std": math.nan,
            }
        else:
            # Global descriptors from RDKit
            try:
                values = calc.CalcDescriptors(mol)
            except Exception:
                # If any descriptor explodes, mark all as NaN for safety
                values = [math.nan] * len(descriptor_names)
            desc_dict = dict(zip(descriptor_names, values))

            # Atom-level aggregates
            agg = compute_atom_level_aggregates(mol)

        desc_dict.update(agg)
        desc_rows.append(desc_dict)

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} molecules...")

    desc_df = pd.DataFrame(desc_rows)

    # 4. Concatenate original data and descriptors, and save
    out_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved descriptors to: {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    # Example usage: adjust 'input.csv' to your actual path
    main("train.csv", "FULL_mfi_rdkit_descriptors.csv")
