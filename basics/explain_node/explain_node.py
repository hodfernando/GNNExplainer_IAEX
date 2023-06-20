import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import CoraGraphDataset
from dgl.nn import GNNExplainer

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