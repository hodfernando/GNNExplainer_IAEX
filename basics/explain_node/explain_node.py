import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import CoraGraphDataset
from dgl.nn import GNNExplainer

import numpy as np
import pandas as pd
from tabulate import tabulate

# Referencia https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.explain.GNNExplainer.html

# Load dataset
data = CoraGraphDataset()
g = data[0]
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']


# Define a model
class Model(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, graph, feat, eweight=None):
        with graph.local_scope():
            feat = self.linear(feat)
            graph.ndata['h'] = feat
            if eweight is None:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            else:
                graph.edata['w'] = eweight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            return graph.ndata['h']


# Train the model
model = Model(features.shape[1], data.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(10):
    logits = model(g, features)
    loss = criterion(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Explain the prediction for node 10
explainer = GNNExplainer(model, num_hops=1)
new_center, sg, feat_mask, edge_mask = explainer.explain_node(10, g, features)
new_center
sg.num_edges()
# Old IDs of the nodes in the subgraph
sg.ndata[dgl.NID]
# Old IDs of the edges in the subgraph
sg.edata[dgl.EID]
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
