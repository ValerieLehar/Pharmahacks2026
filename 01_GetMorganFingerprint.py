import pandas as pd
import numpy as np
import warnings

import rdkit
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Draw, AllChem, PandasTools, BRICS, MACCSkeys, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
#from rdkit.DataStructs.cDatastructs import TanimotoSimilarity, DiceSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, PandasTools, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFreeSASA as fs
from rdkit.Chem.rdmolops import PatternFingerprint
#from rdkit.Chem.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect

## Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

##Reading Data
df = pd.read_csv('./TestData/test_full_warm.csv')


# Creating milecular object from SMiles
PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule')

## Drop invalid molecules
df = df[df['Molecule'].notnull()].reset_index(drop=True)

#Morgan fingerprints
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fingerprints = []

for mol in df['Molecule']:
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fingerprints.append(arr)

#Combine with original Dataframe
MF = pd.concat ([df, pd.DataFrame(fingerprints)], axis = 1)


MF.to_csv('morgan_fingerprints_warmFull.csv', index=False)
np.save('morgan_fingerprints_warmFull.npy', np.array(fingerprints))
#MF.to_excel('morgan_fingerprints.xlsx', index=False)

##print first molecule in dataset
##print(MF.iloc[0,1])
##nmpyrrole = MF.iloc[0,2:]

#Get number of 0s and 1s in the fingerprint
##num_zeros = np.sum(nmpyrrole == 0)
##num_ones = np.sum(nmpyrrole == 1)
##print(f"Number of 0s: {num_zeros}")
##print(f"Number of 1s: {num_ones}")

## TEST
