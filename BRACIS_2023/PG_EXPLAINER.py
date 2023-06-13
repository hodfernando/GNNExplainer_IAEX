import os
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.utils import k_hop_subgraph
from torch_geometric_temporal.signal import temporal_signal_split
from create_dataframe import DatasetLoader
from models.gnn_lstm_model import LSTMGCN
from models.gnn_rnn_model import RecurrentGCN
import torch
from tqdm import tqdm
from pathlib import Path
import pickle
import bz2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Setando e chamando
lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCRN'
reps = 1
train_ratio = 0.8

# def gnnexplainerpredict():
# Diretorios
mydir = os.getcwd()
Path(mydir + '/save/class/').mkdir(parents=True, exist_ok=True)
files_class = os.listdir(mydir + '/save/class/')
results_path = '/results/Explainer/PGExplainer/'
Path(mydir + results_path).mkdir(parents=True, exist_ok=True)

# Carregando Dataset
filename_dataset = f'DATASET_lags_{lags}_datasets_{datasets}_weight_{weight}_out_{out}_'
if filename_dataset in files_class:
    file = bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'rb')
    datasetLoader = pickle.load(file)
    dataset, dataset_standardized = datasetLoader.get_dataset(lags=lags, datasets=datasets,
                                                              weight=weight, out=out)
    file.close()
    print('Dataset carregado pelo pickle data')
else:
    datasetLoader = DatasetLoader()
    dataset, dataset_standardized = datasetLoader.get_dataset(lags=lags, datasets=datasets, weight=weight, out=out)
    with bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'wb') as file:
        pickle.dump(datasetLoader, file)
    file.close()
    print('Dataset carregado e gerado arquivo pickle')

print("Separando dataset em teste e treinamento")

train_dataset, test_dataset = temporal_signal_split(dataset_standardized, train_ratio=train_ratio)

for rep in range(reps):
    # Carregando Model
    if nameModel == 'GCRN':
        model = RecurrentGCN(node_features=lags, filters=128)
    if nameModel == 'GCLSTM':
        model = LSTMGCN(node_features=lags, filters=128)

    epochs = 10
    lr = 0.01
    explanation_type = 'phenomenon'

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type=explanation_type,
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
        # Include only the top 10 most important edges:
        threshold_config=dict(threshold_type='topk', value=10),
    )

    for epoch in tqdm(range(epochs)):
        for time, snapshot in enumerate(train_dataset):
            loss = explainer.algorithm.train(epoch, model=model, x=snapshot.x, edge_index=snapshot.edge_index,
                                             edge_weight=snapshot.edge_attr, target=snapshot.y.unsqueeze(1))

    graph_index = 0
    # Generate the explanation for a particular graph:
    explanation = explainer(x=train_dataset[graph_index].x, edge_index=train_dataset[graph_index].edge_index,
                            edge_weight=train_dataset[graph_index].edge_attr,
                            target=train_dataset[graph_index].y.unsqueeze(1))
    print(f'Generated explanations in {explanation.available_explanations}')
    print(explanation.edge_mask)
    edge_mask = explanation.edge_mask.numpy()

    # Salvar o objeto explainer em um arquivo
    with open(mydir + results_path + f'explainer_{explanation_type}.pkl', 'wb') as file:
        pickle.dump(explainer, file)

    # Salvar o objeto explanation em um arquivo
    with open(mydir + results_path + f'explanation_{explanation_type}.pkl', 'wb') as file:
        pickle.dump(explanation, file)
