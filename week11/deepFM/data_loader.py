import numpy as np
import pandas as pd
import pickle
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            data, self.nrof_emb_categories, self.unique_categories = pickle.load(f)

        self.embedding_columns = ['workclass_cat', 'education_cat', 'marital-status_cat', 'occupation_cat',
                                  'relationship_cat', 'race_cat',
                                  'sex_cat', 'native-country_cat']
        self.nrof_emb_categories = {key + '_cat': val for key, val in self.nrof_emb_categories.items()}
        self.numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                'hours-per-week']

        self.columns = self.embedding_columns + self.numeric_columns

        self.X = data[self.columns].reset_index(drop=True)
        self.y = np.asarray([0 if el == '<50k' else 1 for el in data['salary'].values], dtype=np.int32)

        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        row = self.X.take([idx], axis=0)

        row = {col: torch.tensor(row[col].values, dtype=torch.float32) for i, col in enumerate(self.columns)}

        return row, np.float32(self.y[idx])