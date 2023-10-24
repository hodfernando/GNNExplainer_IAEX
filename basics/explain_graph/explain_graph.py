import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import AvgPooling, GNNExplainer

import numpy as np
import pandas as pd
from tabulate import tabulate

# Referencia https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.explain.GNNExplainer.html

# Load dataset
data = GINDataset('MUTAG', self_loop=True)
dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)


# Define a model
class Model(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.pool = AvgPooling()

    def forward(self, graph, feat, eweight=None):
        with graph.local_scope():
            feat = self.linear(feat)
            graph.ndata['h'] = feat
            if eweight is None:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            else:
                graph.edata['w'] = eweight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            return self.pool(graph, graph.ndata['h'])


# Train the model
feat_size = data[0][0].ndata['attr'].shape[1]
model = Model(feat_size, data.gclasses)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for bg, labels in dataloader:
    logits = model(bg, bg.ndata['attr'])
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Explain the prediction for graph 0
explainer = GNNExplainer(model, num_hops=1)
g, _ = data[0]
features = g.ndata['attr']
feat_mask, edge_mask = explainer.explain_graph(g, features)
feat_mask
edge_mask

# Calcular os valores de std, mean, max e min para feat_mask
feat_mask_std = np.std(np.array(feat_mask))
feat_mask_mean = np.mean(np.array(feat_mask))
feat_mask_max = np.max(np.array(feat_mask))
feat_mask_min = np.min(np.array(feat_mask))

# Calcular os valores de std, mean, max e min para edge_mask
edge_mask_std = np.std(np.array(edge_mask))
edge_mask_mean = np.mean(np.array(edge_mask))
edge_mask_max = np.max(np.array(edge_mask))
edge_mask_min = np.min(np.array(edge_mask))

# Criar o dataframe com os valores
df = pd.DataFrame({'Variable': ['feat_mask', 'edge_mask'],
                   'std': [feat_mask_std, edge_mask_std],
                   'mean': [feat_mask_mean, edge_mask_mean],
                   'max': [feat_mask_max, edge_mask_max],
                   'min': [feat_mask_min, edge_mask_min]})

# Converter o dataframe em tabela LaTeX
table = tabulate(df, headers='keys', tablefmt='latex', floatfmt='.3f')

# Exibir a tabela LaTeX
print(table)
