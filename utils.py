import csv
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
from config import DESCRIPTORS
from sklearn.preprocessing import StandardScaler
from typing import Optional


def read_moleculenet_smiles(data_path: str, target: str, task: str='classification'):
    smiles_data, labels, garbages = [], [], []
    data_name = data_path.split('/')[-1].split('.')[0].upper()
    with open(data_path) as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)

            if mol != None and label != '':
                smiles_data.append(smiles)
                if task == 'classification':
                    labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    raise ValueError('Task Error')

            elif mol is None:
                print(idx)
                garbages.append(smiles)

    print(f'{data_name} | Target : {target}({task})| Total {len(smiles_data)}/{idx+1} instances')
    return smiles_data, labels, garbages

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(f'Total Data Size {data_len}, about to generate scaffolds')
    # for idx, smiles in enumerate(dataset.smiles_data):
    for idx, smiles in enumerate(dataset):
        if idx % log_every_n == 0:
            print("Generating scaffold %d/%d"%(idx, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    scaffolds = {k: sorted(v) for k, v in scaffolds.items()}
    scaffold_sets = [(s, s_set) for (s, s_set) in sorted(scaffolds.items(),
                                                    key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    return scaffold_sets


def scaffold_split(dataset, val_size=0.1, test_size=0.1):
    train_size = 1.0 - val_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    val_cutoff = (train_size + val_size) * len(dataset)

    train_indices, val_indices, test_indices = [], [], []
    train_scaffolds, val_scaffolds, test_scaffolds = [], [], []
    
    print("About to Sort in Scaffold Sets")
    for s, s_set in scaffold_sets:
        if len(train_indices) + len(s_set) > train_cutoff:
            if len(train_indices) + len(val_indices) + len(s_set) > val_cutoff:
                test_indices += s_set
                test_scaffolds += (s, s_set)
            else:
                val_indices += s_set
                val_scaffolds += (s, s_set)
        else:
            train_indices += s_set
            train_scaffolds += (s, s_set)
    return train_indices, val_indices, test_indices, train_scaffolds, val_scaffolds, test_scaffolds

def smiles_to_df_with_descriptors(smiles, label):
    df = pd.DataFrame()
    df['smiles'] = smiles
    features = []
    for feature in DESCRIPTORS:
        df[feature] = df['smiles'].apply(lambda x: getattr(Descriptors, feature)(Chem.MolFromSmiles(x)))
        features.append(feature)
    
    label = list(map(lambda x: x if x == 1 else -1, label))
    df['label'] = label
    df.fillna(df.mean(), inplace=True)
    return df

def smiles_to_df_with_fingerprints(smiles, label):
    df = pd.DataFrame()
    df['smiles'] = smiles
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fingerprints = [list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)) for mol in mols]
    df2 = pd.DataFrame(fingerprints, columns=[f'X{i:04d}' for i in range(len(fingerprints[0]))])
    df_total = pd.concat([df, df2], axis=1)
    df_total['label'] = label
    return df_total