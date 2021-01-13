import numpy as np
import pandas as pd

import torch
import torch.nn as nn

class DenseFeatureLayer(nn.Module):

    def __init__(self, input_size, nrof_cat, emb_dim,
                 emb_columns, numeric_columns):
        super(DenseFeatureLayer, self).__init__()

        self.emb_columns = emb_columns
        self.numeric_columns = numeric_columns

        self.embeddings = {}
        for i, col in enumerate(self.emb_columns):
            self.embeddings[col] = torch.nn.Embedding(nrof_cat[col], emb_dim)

        self.feature_bn = torch.nn.BatchNorm1d(input_size)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        numeric_feats = torch.tensor(pd.DataFrame(input_data)[self.numeric_columns].values, dtype=torch.float32)

        emb_output = None
        for i, col in enumerate(self.emb_columns):
            if emb_output is None:
                emb_output = self.embeddings[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))
            else:
                emb_output = torch.cat(
                    [emb_output,
                     self.embeddings[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))],
                    dim=1)

        concat_input = torch.cat([numeric_feats, emb_output], dim=1)
        output = self.feature_bn(concat_input)

        return output

class GLULayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(GLULayer, self).__init__()

        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc_bn = torch.nn.BatchNorm1d(output_size)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        output = self.fc(input_data)
        output = self.fc_bn(output)
        output = torch.nn.functional.glu(output)

        return output

class FeatureTransformer(nn.Module):

    def __init__(self, nrof_glu, input_size, output_size):
        super(FeatureTransformer, self).__init__()

        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

        self.nrof_glu = nrof_glu
        self.glu_layers = []

        for i in range(nrof_glu):
            self.glu_layers.append(GLULayer(input_size[i], output_size))

    def forward(self, input_data):
        layer_input_data = input_data
        for i in range(self.nrof_glu):
            layer_input_data = torch.add(layer_input_data, self.glu_layers[i](layer_input_data))
            layer_input_data = layer_input_data * self.scale



class AttentiveTransformer(nn.Module):

    def __init__(self):
        super(AttentiveTransformer, self).__init__()


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):

        return output

class TabNet(nn.Module):
    def __init__(self):
        super(TabNet, self).__init__()




    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):

        return