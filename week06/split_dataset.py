import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = pd.read_csv('../week01/data/X_cat.csv', sep = '\t', index_col=0)
target = pd.read_csv('../week01/data/y_cat.csv', sep = '\t', index_col=0, names=['status'])

target = target.iloc[:, :]
target[target == 'Died'] = 'Euthanasia'

le = LabelEncoder()
y = le.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :].values, y,
                                                    test_size=0.2, stratify=y, random_state=42)

print('Size of train set: ', len(X_train))
print('Size of test set: ', len(X_test))

with open('./data/X_train_cat.pickle', 'wb') as f:
    pickle.dump((X_train, y_train), f)

with open('./data/X_test_cat.pickle', 'wb') as f:
    pickle.dump((X_test, y_test), f)