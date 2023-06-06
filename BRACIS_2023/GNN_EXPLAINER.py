import os
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer
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

        # for explanation_type in ['phenomenon', 'model']:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=10),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            ),
        )

        # SP/SP - Sudeste - 3550308
        # Manaus/AM - Norte - 1302603
        # Fortaleza/CE - Nordeste - 2304400
        # Brasília/DF - Centro - 5300108
        # Florianópolis/SC - Sul - 4205407
        node_indices = datasetLoader._label_encoder_CODMUNDV.transform([3550308, 1302603, 5300108])
        targets, preds, results = [], [], []
        mae, mse, rmse, r2 = 0, 0, 0, 0
        for node_index in tqdm(node_indices, leave=False, desc='Test Explainer'):
            for time, snapshot in enumerate(test_dataset):
                target = snapshot.y  # if explanation_type == 'phenomenon' else None
                explanation = explainer(x=snapshot.x, edge_index=snapshot.edge_index, index=node_index,
                                        target=target, edge_weight=snapshot.edge_attr)

                subset, _, _, _ = k_hop_subgraph(int(node_index), num_hops=1, edge_index=snapshot.edge_index)

                targets.append(snapshot.subgraph(subset).y)
                preds.append(explanation.subgraph(subset).target.squeeze())

                mae += mean_absolute_error(targets[time], preds[time])
                mse += mean_squared_error(targets[time], preds[time])
                rmse += np.sqrt(mse)
                r2 += r2_score(targets[time], preds[time])
            mae /= time
            mse /= time
            rmse /= time
            r2 /= time
            # print(f"Node: {node_index}")
            # print(f"MAE: {mae}")
            # print(f"MSE: {mse}")
            # print(f"RMSE: {rmse}")
            # print(f"R²: {r2}")

            result = {
                'Node': node_index,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            }
            results.append(result)

        df_results = pd.DataFrame(results)
        print(df_results)

        # nodes = np.arange(significance.shape[0])
        #
        # # Gráfico de dispersão
        # plt.figure(figsize=(10, 6))
        # plt.scatter(nodes, significance)
        # plt.xlabel('Nós')
        # plt.ylabel('Significância')
        # plt.title(f'Gráfico de Dispersão - Significância dos Nós em relação ao Nó {node_idx}')
        # plt.show()


# Setuando e chamando
lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCRN'
reps = 1
train_ratio = 0.8
gnnexplainerpredict()
