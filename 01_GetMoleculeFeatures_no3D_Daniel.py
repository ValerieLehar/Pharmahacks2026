import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Draw, AllChem, PandasTools, BRICS, MACCSkeys, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
#from rdkit.DataStructs.cDatastructs import TanimotoSimilarity, DiceSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFreeSASA as fs
from rdkit.ML.Descriptors import MoleculeDescriptors
from multiprocessing import Pool, cpu_count

df = pd.read_csv('train.csv')

mol = Chem.MolFromSmiles(df.iloc[1]['SMILES'])

# Precompute once
descriptor_names = [name for name, _ in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def extract_all_mol_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    # --- 2D descriptors ---
    values = calculator.CalcDescriptors(mol)
    all_properties = dict(zip(descriptor_names, values))
    
    return all_properties


# Parallel execution

if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        results = p.map(extract_all_mol_data, df['SMILES'])

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("molecular_descriptors.csv", index=False)
