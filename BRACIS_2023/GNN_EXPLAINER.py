import os
import numpy as np
from torch_geometric.contrib.explain import PGMExplainer
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric_temporal.signal import temporal_signal_split
from create_dataframe import DatasetLoader
from models.gnn_lstm_model import LSTMGCN
from models.gnn_rnn_model import RecurrentGCN
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import bz2


def gnnexplainerpredict():
    # Diretorios
    mydir = os.getcwd()
    Path(mydir + '/save/class/').mkdir(parents=True, exist_ok=True)
    files_class = os.listdir(mydir + '/save/class/')

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
        file = bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'wb')
        # pickle.dump(datasetLoader, file)
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

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # adam aleatorio?

        model.train()
        print("Treinamento")
        time = 0
        for epoch in tqdm(range(100)):
            cost = 0
            for time, snapshot in enumerate(train_dataset):  # Shuffle
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                cost = torch.mean((y_hat - snapshot.y) ** 2)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()

        print("Fim do treinamento")

        print("Teste")

        model.eval()

        explainer = Explainer(
            model=model, algorithm=PGMExplainer(), node_mask_type='attributes',
            explanation_type='phenomenon',
            model_config=ModelConfig(mode='regression',
                                     task_level='node', return_type='raw'))

        time = 0
        for time, snapshot in enumerate(test_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            break

        node_idx = 0

        explanation = explainer(x=snapshot.x, edge_index=snapshot.edge_index, index=node_idx,
                                target=y_hat, edge_weight=snapshot.edge_weight)

        significance = explanation.pgm_stats
        print(f'Significance of relevant neighbors: {significance}')

        nodes = np.arange(significance.shape[0])

        # Gráfico de dispersão
        plt.figure(figsize=(10, 6))
        plt.scatter(nodes, significance)
        plt.xlabel('Nós')
        plt.ylabel('Significância')
        plt.title(f'Gráfico de Dispersão - Significância dos Nós em relação ao Nó {node_idx}')
        plt.show()


# Setuando e chamando
lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCRN'
reps = 1
train_ratio = 0.8
gnnexplainerpredict()
