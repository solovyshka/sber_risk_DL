import numpy as np
import pandas as pd
import pickle

embedding_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                 'sex', 'native-country']
nrof_emb_categories = {}
unique_categories = {}

with open('/home/firiuza/sber_risk_DL/week11/data/train_adult.pickle', 'rb') as f:
    data, _, _ = pickle.load(f)

for cat in embedding_columns:
    nrof_unique = np.unique(data[cat].values.astype(np.str))
    # data.groupby(cat).agg({cat: 'count'})
    unique_categories[cat] = nrof_unique
    nrof_emb_categories[cat] = len(nrof_unique)
    data[cat + '_cat'] = [np.where(nrof_unique == val)[0][0] for i, val in enumerate(data[cat].values.astype(np.str))]

with open('/home/firiuza/sber_risk_DL/week11/data/train_adult.pickle', 'wb') as f:
    pickle.dump([data, nrof_emb_categories, unique_categories], f)


with open('/home/firiuza/sber_risk_DL/week11/data/valid_adult.pickle', 'rb') as f:
    data, _, _ = pickle.load(f)

for cat in embedding_columns:
    data[cat + '_cat'] = [np.where(unique_categories[cat] == val)[0][0] for i, val in enumerate(data[cat].values.astype(np.str))]

with open('/home/firiuza/sber_risk_DL/week11/data/valid_adult.pickle', 'wb') as f:
    pickle.dump([data, nrof_emb_categories, unique_categories], f)