import pandas as pd
import pickle
import numpy as np

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate

BATCH_SIZE = 128
EPOCHS = 100

class CustomDataset(Dataset):
    # Конструктор, где считаем датасет
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.X, self.target = pickle.load(f)

        return

    def __len__(self):
        return len(self.X)

    # Переопределяем метод,
    # который достает по индексу наблюдение из датасет
    def __getitem__(self, idx):
        return self.X[idx], self.target[idx]


class CustomSampler(Sampler):

    # Конструктор, где инициализируем индексы элементов
    def __init__(self, data):
        self.data_indices = np.arange(len(data))

        shuffled_indices = np.random.permutation(len(self.data_indices))

        self.data_indices = np.ascontiguousarray(self.data_indices)[shuffled_indices]

        return

    def __len__(self):
        return len(self.data_indices)

    # Возращает итератор,
    # который будет возвращать индексы из перемешанного датасета
    def __iter__(self):
        return iter(self.data_indices)


def collate(batch):
    return default_collate(batch)


def create_data_loader(train_dataset, train_sampler,
                       test_dataset, test_sampler):
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler,
                              batch_size=BATCH_SIZE, collate_fn=collate,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler,
                             batch_size=BATCH_SIZE, collate_fn=collate,
                             shuffle=False)

    return train_loader, test_loader


# Создаем объекты Custom Dataset и Sampler
train_ds = CustomDataset('./data/X_train_cat.pickle')
train_sampler = CustomSampler(train_ds.X)

test_ds = CustomDataset('./data/X_test_cat.pickle')
test_sampler = CustomSampler(test_ds.X)

train_loader, test_loader = create_data_loader(train_ds, train_sampler,
                                               test_ds, test_sampler)

def run_train():
    print('Run train')
    for epoch in range(EPOCHS):
        for features, labels in train_loader:
            print(features, labels)

        # Run validation
        print('Run validation')
        for features, labels in test_loader:
            print(features, labels)

    return

run_train()

