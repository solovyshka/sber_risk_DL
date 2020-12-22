import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class DenseFeatureLayer(nn.Module):

    def __init__(self, nrof_cat, emb_dim,
                 emb_columns, numeric_columns):
        super(DenseFeatureLayer, self).__init__()

        self.emb_columns = emb_columns
        self.numeric_columns = numeric_columns

        self.numeric_feature_bn = torch.nn.BatchNorm1d(len(numeric_columns))

        input_size = len(emb_columns) + len(numeric_columns)
        self.first_feature_bn = torch.nn.BatchNorm1d(input_size)

        input_size = (len(emb_columns) + len(numeric_columns)) * emb_dim
        self.second_feature_bn = torch.nn.BatchNorm1d(input_size)

        self.first_order_embd = {}
        for i, col in enumerate(self.emb_columns):
            self.first_order_embd[col] = torch.nn.Embedding(nrof_cat[i], 1)
        for i, col in enumerate(numeric_columns):
            self.first_order_embd[col] = torch.tensor(requires_grad=True)

        self.second_order_embd = {}
        for i, col in enumerate(self.emb_columns):
            self.second_order_embd[col] = torch.nn.Embedding(nrof_cat[i], emb_dim)
        for i, col in enumerate(numeric_columns):
            self.second_order_embd[col] = torch.tensor(requires_grad=True)

        return

    def forward(self, input_data):
        numeric_features = torch.stack([input_data[col] for col in self.numeric_columns], dim=1)
        numeric_features = self.numeric_feature_bn(numeric_features)

        first_order_embd_output = None
        for i, col in enumerate(self.emb_columns):
            if first_order_embd_output is None:
                first_order_embd_output = self.first_order_embd[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))
            else:
                first_order_embd_output = torch.cat(
                    [first_order_embd_output, self.first_order_embd[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))],
                    dim=1)

        for i, col in enumerate(self.numeric_columns):
            if first_order_embd_output is None:
                first_order_embd_output = torch.mul(numeric_features[i], self.first_order_embd[col])
            else:
                first_order_embd_output = torch.cat(
                    [first_order_embd_output,
                     torch.mul(numeric_features[i], self.first_order_embd[col])],
                    dim=1)

        second_order_embd_output = None
        for i, col in enumerate(self.emb_columns):
            if second_order_embd_output is None:
                second_order_embd_output = self.first_order_embd[col](
                    torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))
            else:
                second_order_embd_output = torch.cat(
                    [second_order_embd_output,
                     self.first_order_embd[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))],
                    dim=1)

        for i, col in enumerate(self.numeric_columns):
            if second_order_embd_output is None:
                second_order_embd_output = torch.mul(numeric_features[i], self.first_order_embd[col])
            else:
                second_order_embd_output = torch.cat(
                    [second_order_embd_output,
                     torch.mul(numeric_features[i], self.first_order_embd[col])],
                    dim=1)

        first_order_embd_output = self.first_feature_bn(first_order_embd_output)
        second_order_embd_output = self.first_feature_bn(second_order_embd_output)

        return first_order_embd_output, second_order_embd_output

class FMLayer(nn.Module):

    def __init__(self, ):
        super(FMLayer, self).__init__()

        return

    def forward(self, first_order_embd, second_order_embd):
        # sum_square part
        summed_features_embd = torch.sum(second_order_embd, dim=1)  # None * K
        summed_features_embd_square = torch.square(summed_features_embd)  # None * K

        # square_sum part
        squared_features_embd = torch.square(second_order_embd)
        squared_sum_features_embd = torch.sum(squared_features_embd, dim=1)  # None * K

        # second order
        second_order = 0.5 * torch.sub(summed_features_embd_square,
                                       squared_sum_features_embd)  # None * K

        return first_order_embd, second_order

class MLPLayer(nn.Module):

    def __init__(self, input_size, nrof_layers, nrof_neurons, output_size):
        super(MLPLayer, self).__init__()
        self.deep_block = torch.nn.Sequential((torch.nn.Linear(input_size[i], nrof_neurons[i]),
                                               torch.nn.ReLU(),
                                               torch.nn.BatchNorm1d(nrof_neurons[i]))
                                               for i in range(nrof_layers - 1))

        self.output_layer = torch.nn.Linear(nrof_neurons[-1], output_size)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        output = self.deep_block(input_data)
        output = self.output_layer(output)

        return output


class DeepFMNet(nn.Module):

    def __init__(self, nrof_cat, emb_dim,
                 emb_columns, numeric_columns,
                 nrof_layers, nrof_neurons, output_size, nrof_out_classes):
        super(DeepFMNet, self).__init__()

        self.features_embd = DenseFeatureLayer(nrof_cat, emb_dim, emb_columns, numeric_columns)
        self.FM = FMLayer()

        input_size = (len(emb_columns) + len(numeric_columns)) * emb_dim
        self.MLP = MLPLayer(input_size, nrof_layers, nrof_neurons, output_size)

        input_size = len(emb_columns) + len(numeric_columns) + emb_dim + output_size
        self.dense_layer = nn.Linear(input_size, nrof_out_classes)

    def forward(self, input_data):
        first_order_embd, second_order_embd = self.features_embd(input_data)
        FM_first_order, FM_second_order = self.FM(first_order_embd, second_order_embd)
        Deep = self.MLP(second_order_embd)

        concat_output = torch.cat([FM_first_order, FM_second_order, Deep], dim=1)
        output = self.dense_layer(concat_output)

        return output
