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




df = pd.read_csv('small_data.csv')


import time
def extract_all_mol_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
   
    # --- 2D Properties ---
    all_properties = {}
    for name, func in Descriptors._descList:
        try:
            all_properties[name] = func(mol)
        except Exception:
            all_properties[name] = None
           
    # --- 3D Properties ---
    try:
        mol3d = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol3d, AllChem.ETKDG()) == 0:
            AllChem.UFFOptimizeMolecule(mol3d)
            desc_3d = Descriptors3D.CalcMolDescriptors3D(mol3d)
            all_properties.update(desc_3d)
        else:
            # If 3D fails, fill 3D keys with None
            pass
    except Exception:
        pass
       
    return all_properties
start_time = time.perf_counter()
extract_all_mol_data(df.iloc[1]['SMILES'])
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")
