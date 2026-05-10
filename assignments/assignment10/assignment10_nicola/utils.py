import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd
from rdkit.Chem import Descriptors


def maccs_fp_from_smiles(smiles_list):
    fps = []
    valid_idx = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(np.array(fp))
        valid_idx.append(i)

    return np.array(fps), valid_idx

def maccs_fp_from_smiles_as_bitvectors(smiles_list):
    fps = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(fp)
    return fps

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol), 
        "EState_VSA5": Descriptors.EState_VSA5(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHAcc": Descriptors.NumHAcceptors(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Descriptors.RingCount(mol),
        "qed": Descriptors.qed(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NOCount": Descriptors.NOCount(mol),
    })