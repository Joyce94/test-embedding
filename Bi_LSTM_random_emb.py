import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)

        self.word_embeddings = nn.Embedding(args.embed_num, args.embedding_dim, padding_idx=args.padID)
        self.rand_init(self.word_embeddings)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, bidirectional=True, dropout=args.dropout_model)

        self.hidden2label1 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.hidden2label2 = nn.Linear(args.hidden_dim, args.class_num)
        self.hidden = self.init_hidden(args.batch_size, args.use_cuda)


    def rand_init(self, input_emb):
        print("Use uniform to initialize the embedding")
        input_emb.weight.data.uniform_(-0.01, 0.01)
        # print(input_emb.padding_idx)
        if input_emb.padding_idx is not None:
            input_emb.weight.data[input_emb.padding_idx].fill_(0)

    def init_hidden(self, batch_size, use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(2, batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(2, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, batch_size, self.hidden_dim)))

    def forward(self, sentence):
        # print(sentence)                                     # [torch.LongTensor of size 42x64]
        x = self.word_embeddings(sentence)

        x = self.dropout_embed(x)

        lstm_out, self.hidden = self.lstm(x, self.hidden)  # lstm_out 10*5*50 hidden 1*5*50 *2

        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2))
        # print(lstm_out.size())
        lstm_out = lstm_out.squeeze(2)

        y = self.hidden2label1(F.tanh(lstm_out))
        y = self.hidden2label2(F.tanh(y))

        log_probs = y
        return log_probs
