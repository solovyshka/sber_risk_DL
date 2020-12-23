import torch
print(torch.__version__)

import torch.optim as optim
import torch.utils.data as data_utils

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.metrics import Accuracy

from week11.deepFM.network import DeepFMNet
from week11.deepFM.data_loader import CustomDataset

EPOCHS = 500
EMBEDDING_SIZE = 5
BATCH_SIZE = 512
NROF_LAYERS = 3
NROF_NEURONS = 50
DEEP_OUTPUT_SIZE = 50
NROF_OUT_CLASSES = 1
LEARNING_RATE = 3e-4
TRAIN_PATH = '/home/firiuza/sber_risk_DL/week11/data/train_adult.pickle'
VALID_PATH = '/home/firiuza/sber_risk_DL/week11/data/valid_adult.pickle'

class DeepFM:
    def __init__(self):
        self.train_dataset = CustomDataset(TRAIN_PATH)
        self.train_loader = data_utils.DataLoader(dataset=self.train_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=True)

        self.build_model()

        self.log_params()

        self.train_writer = SummaryWriter('./logs/train')
        self.valid_writer = SummaryWriter('./logs/valid')

        return

    def build_model(self):
        self.network = DeepFMNet(nrof_cat=self.train_dataset.nrof_emb_categories, emb_dim=EMBEDDING_SIZE,
                                 emb_columns=self.train_dataset.embedding_columns,
                                 numeric_columns=self.train_dataset.numeric_columns,
                                 nrof_layers=NROF_LAYERS, nrof_neurons=NROF_NEURONS,
                                 output_size=DEEP_OUTPUT_SIZE,
                                 nrof_out_classes=NROF_OUT_CLASSES)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy()
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        return

    def log_params(self):
        return

    def load_model(self, restore_path=''):
        if restore_path == '':
            self.step = 0
        else:
            pass

        return

    def run_train(self):
        print('Run train ...')

        self.load_model()

        for epoch in range(EPOCHS):
            self.network.train()

            for features, label in self.train_loader:
                # Reset gradients
                self.optimizer.zero_grad()

                output = self.network(features)
                # Calculate error and backpropagate
                loss = self.loss(output, label)

                output = torch.sigmoid(output)

                loss.backward()
                acc = self.accuracy(output, label).item()

                # Update weights with gradients
                self.optimizer.step()

                self.train_writer.add_scalar('CrossEntropyLoss', loss, self.step)
                self.train_writer.add_scalar('Accuracy', acc, self.step)

                self.step += 1

                if self.step % 50 == 0:
                    print('EPOCH %d STEP %d : train_loss: %f train_acc: %f' %
                          (epoch, self.step, loss.item(), acc))

            # self.train_writer.add_histogram('hidden_layer', self.network.linear1.weight.data, self.step)

            # Run validation
            #TODO

        return


deep_fm = DeepFM()
deep_fm.run_train()