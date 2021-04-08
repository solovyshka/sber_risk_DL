import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):

    def __init__(self, input_size, output_size=64):
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.coeff = np.sqrt(output_size)

        self.W_q = nn.Linear(input_size, output_size)
        self.W_k = nn.Linear(input_size, output_size)
        self.W_v = nn.Linear(input_size, output_size)

        self.softmax = nn.Softmax()

        return

    def forward(self, embeddings):
        # [batch_size, nrof_words, embd_dim]
        # [[1, 2, 3],
        #  [4, 5, 6]]
        query = self.W_q(embeddings)
        key = self.W_k(embeddings)
        value = self.W_v(embeddings)

        # dim(key) == dim(query) == dim(value) == [batch_size, nrof_words, output_size]
        k_tr = torch.transpose(key, 2, 1)
        score = torch.matmul(query, k_tr)

        score = torch.div(score, self.coeff) # score / self.coeff

        score = self.softmax(score)

        output = torch.mul(score, value)

        return output

#Toy example
test_sample = torch.FloatTensor([[
    [1, 2, 3], [4, 5, 6], [2, 3, 6]],
    [[7, 2, 1], [8, 5, 2], [0, 0, 0]]])

word_embd_dim = 3
output_size = 3
attention = SelfAttention(word_embd_dim, output_size)
attention(test_sample)