import torch
import torch.nn.functional as functional
from torch_geometric_temporal.nn.recurrent import GConvGRU


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters, K=2, normalization="sym"):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(in_channels=node_features, out_channels=filters, K=K, normalization=normalization,
                                  bias=True)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = functional.relu(h)
        h = self.linear(h)
        return h
