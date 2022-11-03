import argparse
from config import TARGET_DICT, DATA_PATH, TASK
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import random
import numpy as np
from rdkit import RDLogger
import warnings
from utils import read_moleculenet_smiles, scaffold_split, smiles_to_df_with_descriptors
from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.decomposition import PCA
import pandas as pd


warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

def get_parser():
    parser = argparse.ArgumentParser(description= 'Kernel SVM with Molecular Descriptors')
    parser.add_argument('--d_name', default='BACE', type=str, choices=list(TARGET_DICT.keys()))
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--f-name', default='result.csv', type=str)
    return parser

def main(args):
    assert args.d_name in list(TARGET_DICT.keys())
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_name = args.d_name
    target = TARGET_DICT[data_name][0]
    data_path = DATA_PATH[data_name]
    task = TASK[data_name]

    param_grid = {'kernel': ['rbf'],
                  'C': [0.1, 0.5, 1, 5, 10],
                  'gamma': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}
    if task =='classification':
        param_grid['class_weight'] = ['balanced', None]
    
    ##Load Data
    smiles_data, labels, garbages = read_moleculenet_smiles(data_path, target, task)
    train_indices, val_indices, test_indices, train_scaffolds, val_scaffolds, test_scaffolds = scaffold_split(smiles_data, val_size=0.1, test_size=0.1)

    train_smiles, val_smiles, test_smiles = [list(map(smiles_data.__getitem__, indices)) for indices in [train_indices, val_indices, test_indices]]
    train_labels, val_labels, test_labels = [list(map(labels.__getitem__, indices)) for indices in [train_indices, val_indices, test_indices]]
    
    train_df = smiles_to_df_with_descriptors(train_smiles, train_labels)
    val_df = smiles_to_df_with_descriptors(val_smiles, val_labels)
    test_df = smiles_to_df_with_descriptors(test_smiles, test_labels)
    
    train_x, train_y = np.array(train_df.drop(['smiles', 'label'], axis=1)), np.array(train_df['label'])
    val_x, val_y = np.array(val_df.drop(['smiles', 'label'], axis=1)), np.array(val_df['label'])
    test_x, test_y = np.array(test_df.drop(['smiles', 'label'], axis=1)), np.array(test_df['label'])
    
    ##Dimensionality Reduction
    pca = PCA(n_components=50)
    train_x = pca.fit_transform(train_x)
    val_x = pca.fit_transform(val_x)
    test_x = pca.fit_transform(test_x)
    
    ##Model Training
    base_estimator = getattr(svm, 'SVC' if task == 'classification' else 'SVR')()
    
    split_index= [-1] * len(train_x) + [0] * len(val_x)
    X = np.concatenate((train_x, val_x), axis=0)
    Y = np.concatenate((train_y, val_y), axis=0)
    pds = PredefinedSplit(test_fold=split_index)
    scoring = 'roc_auc' if task == 'classification' else 'neg_root_mean_squared_error'
    cv_estimator = GridSearchCV(base_estimator, param_grid, cv=pds, scoring=scoring)
    cv_estimator.fit(X, Y)
    
    ##Test
    records = cv_estimator.best_params_
    records['Data'] = data_name
    if task == 'classification':
        records['Metric'] = 'ROC-AUC'
        records['Score'] = cv_estimator.score(test_x, test_y)
    else:
        records['Metric'] = 'RMSE'
        records['Score'] = cv_estimator.score(test_x, test_y) * -1
        records['class_weight'] = 'X'
    records['Representation'] = 'Descriptors'
    records = {k:records[k] for k in ['Data', 'Representation', 'kernel', 'class_weight', 'C', 'gamma', 'Metric', 'Score']}
    
    df = pd.DataFrame.from_records([records])
    if not os.path.exists(args.f_name):
        df.to_csv(os.path.join(args.f_name), mode='w', index=False,)
    else:
        df.to_csv(os.path.join(args.f_name), mode='a', index=False, header=False)
        
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    for d in TARGET_DICT.keys():
        args.d_name = d
        main(args)