import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch

print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.metrics import Accuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_SIZE = 36
HIDDEN_SIZE = 25
OUTPUT_SIZE = 5
LEARNING_RATE = 1e-2
EPOCHS = 400
BATCH_SIZE = 256
EMBEDDING_SIZE = 5


class CustomDataset(Dataset):
    # Конструктор, где считаем датасет
    def __init__(self):
        X = pd.read_csv('./data/X_cat.csv', sep='\t', index_col=0)
        target = pd.read_csv('./data/y_cat.csv', sep='\t', index_col=0, names=['status'])  # header=-1,

        weekday_columns = ['Weekday_0', 'Weekday_1', 'Weekday_2',
                           'Weekday_3', 'Weekday_4', 'Weekday_5', 'Weekday_6']
        weekdays = np.argmax(X[weekday_columns].values, axis=1)

        X.drop(weekday_columns, axis=1, inplace=True)

        X['Weekday_cos'] = np.cos(2 * np.pi / 7.) * weekdays
        X['Weekday_sin'] = np.sin(2 * np.pi / 7.) * weekdays

        X['Hour_cos'] = np.cos(2 * np.pi / 24.) * X['Hour'].values
        X['Hour_sin'] = np.sin(2 * np.pi / 24.) * X['Hour'].values

        X['Month_cos'] = np.cos(2 * np.pi / 12.) * X['Month'].values
        X['Month_sin'] = np.sin(2 * np.pi / 12.) * X['Month'].values

        X['Gender'] = np.argmax(X[['Sex_Female', 'Sex_Male', 'Sex_Unknown']].values, axis=1)

        X.drop(['Sex_Female', 'Sex_Male', 'Sex_Unknown'], axis=1, inplace=True)

        print(X.shape)
        print(X.head())

        target = target.iloc[:, :].values
        target[target == 'Died'] = 'Euthanasia'

        le = LabelEncoder()
        self.y = le.fit_transform(target)

        self.X = X.values

        self.columns = X.columns.values

        self.embedding_column = 'Gender'
        self.nrof_emb_categories = 3
        self.numeric_columns = ['IsDog', 'Age', 'HasName', 'NameLength', 'NameFreq', 'MixColor', 'ColorFreqAsIs',
                                'ColorFreqBase', 'TabbyColor', 'MixBreed', 'Domestic', 'Shorthair', 'Longhair',
                                'Year', 'Day',  'Breed_Chihuahua Shorthair Mix', 'Breed_Domestic Medium Hair Mix',
                                'Breed_Domestic Shorthair Mix', 'Breed_German Shepherd Mix', 'Breed_Labrador Retriever Mix',
                                 'Breed_Pit Bull Mix', 'Breed_Rare',
                                'SexStatus_Flawed', 'SexStatus_Intact', 'SexStatus_Unknown',
                                'Weekday_cos', 'Weekday_sin', 'Hour_cos', 'Hour_sin',
                                'Month_cos', 'Month_sin']

        return

    def __len__(self):
        return len(self.X)

    # Переопределяем метод,
    # который достает по индексу наблюдение из датасет
    def __getitem__(self, idx):

        row = self.X[idx, :]

        row = {col: torch.tensor(row[i]) for i, col in enumerate(self.columns)}

        return row, self.y[idx]

class MLPNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, nrof_cat, emb_dim,
                 emb_columns, numeric_columns):
        super(MLPNet, self).__init__()
        self.emb_columns = emb_columns
        self.numeric_columns = numeric_columns

        self.emb_layer = torch.nn.Embedding(nrof_cat, emb_dim)

        self.feature_bn = torch.nn.BatchNorm1d(input_size)

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear1.apply(self.init_weights)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)

        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2.apply(self.init_weights)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)

        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, x):
        emb_output = self.emb_layer(torch.tensor(x[self.emb_columns], dtype=torch.int64))
        numeric_feats = torch.tensor(pd.DataFrame(x)[self.numeric_columns].values, dtype=torch.float32)

        concat_input = torch.cat([numeric_feats, emb_output], dim=1)
        output = self.feature_bn(concat_input)

        output = self.linear1(output)
        output = self.bn1(output)
        output = torch.relu(output)

        output = self.linear2(output)
        output = self.bn2(output)
        output = torch.relu(output)

        output = self.linear3(output)
        predictions = torch.softmax(output, dim=1)

        return predictions


def run_train(model, train_loader):
    step = 0
    for epoch in range(EPOCHS):
        model.train()

        for features, label in train_loader:
            # Reset gradients
            optimizer.zero_grad()

            output = model(features)
            # Calculate error and backpropagate
            loss = criterion(output, label)
            loss.backward()
            acc = accuracy(output, label).item()

            # Update weights with gradients
            optimizer.step()

            step += 1

            if step % 100 == 0:
                print('EPOCH %d STEP %d : train_loss: %f train_acc: %f' %
                      (epoch, step, loss.item(), acc))


    return step

animal_dataset = CustomDataset()
train_loader = data_utils.DataLoader(dataset=animal_dataset,
                                     batch_size=BATCH_SIZE, shuffle=True)

model = MLPNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, animal_dataset.nrof_emb_categories,
               EMBEDDING_SIZE,
               animal_dataset.embedding_column, animal_dataset.numeric_columns)

criterion = nn.CrossEntropyLoss()
accuracy = Accuracy()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

step = run_train(model, train_loader)