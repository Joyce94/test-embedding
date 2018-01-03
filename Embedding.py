import torch.nn as nn

class Embedding(nn.Embedding):
    def reset_parameters(self):
        print("Use uniform to initialize the embedding")
        # self.weight.data.normal_(0, 1)
        # if self.padding_idx is not None:
        #     self.weight.data[self.padding_idx].fill_(0)

        self.weight.data.uniform_(-0.01, 0.01)
        # print(self.padding_idx)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class ConstEmbedding(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(ConstEmbedding, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx, sparse=True)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=False)

    def cuda(self, device_id=None):
        """
           The weights should be always on cpu
       """
        return self._apply(lambda t: t.cpu())

    def forward(self, input):
        """
           return cpu tensor
       """
        # is_cuda = next(input).is_cuda
        is_cuda = input.is_cuda
        if is_cuda:
            input = input.cpu()
        self.embedding._apply(lambda t: t.cpu())

        x = self.embedding(input)
        if is_cuda: x = x.cuda()

        return x









