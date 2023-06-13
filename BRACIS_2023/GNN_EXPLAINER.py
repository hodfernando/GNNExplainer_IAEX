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
results_path = '/results/Explainer/GNNExplainer/'
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
    results = []
    for explanation_type in ['phenomenon', 'model']:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type=explanation_type,
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
        for node_index in tqdm(node_indices, leave=False, desc='Test Explainer'):
            targets, preds = [], []
            mae, mse, rmse, r2 = 0, 0, 0, 0
            for time, snapshot in enumerate(test_dataset):
                target = snapshot.y.unsqueeze(1) if explanation_type == 'phenomenon' else None
                explanation = explainer(x=snapshot.x, edge_index=snapshot.edge_index, index=node_index,
                                        target=target, edge_weight=snapshot.edge_attr)

                subset, edge_index, mapping, edge_mask = k_hop_subgraph(int(node_index), num_hops=1,
                                                                        edge_index=snapshot.edge_index)
                targets.append((snapshot.subgraph(
                    subset).y * datasetLoader.std_stacked_dataset[subset]) + datasetLoader.mean_stacked_dataset[subset])
                preds.append((explanation.subgraph(
                    subset).target.squeeze() * datasetLoader.std_stacked_dataset[subset]) +
                             datasetLoader.mean_stacked_dataset[subset])
                # try:
                #     explanation.visualize_graph('subgraph.png')
                #     explanation.visualize_feature_importance('feature_importance.png', top_k=10)

                mae += mean_absolute_error(targets[time], preds[time])
                mse += mean_squared_error(targets[time], preds[time])
                rmse += np.sqrt(mse)
                r2 += r2_score(targets[time], preds[time])
            mae /= (time + 1)
            mse /= (time + 1)
            rmse /= (time + 1)
            r2 /= (time + 1)

            result = {
                'Explanation': explanation_type,
                'Node': node_index,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            }
            results.append(result)

            # Salvar o objeto explainer em um arquivo
            with open(mydir + results_path + f'explainer_{explanation_type}_{node_index}.pkl', 'wb') as file:
                pickle.dump(explainer, file)

            # Salvar o objeto explanation em um arquivo
            with open(mydir + results_path + f'explanation_{explanation_type}_{node_index}.pkl', 'wb') as file:
                pickle.dump(explanation, file)

    df_results = pd.DataFrame(results)
    print()
    print(df_results)
    df_results.to_csv(mydir + results_path + 'explanation.csv')

    # nodes = np.arange(significance.shape[0])
    #
    # # Gráfico de dispersão
    # plt.figure(figsize=(10, 6))
    # plt.scatter(nodes, significance)
    # plt.xlabel('Nós')
    # plt.ylabel('Significância')
    # plt.title(f'Gráfico de Dispersão - Significância dos Nós em relação ao Nó {node_idx}')
    # plt.show()

# # Carregar o objeto explainer do arquivo
# with open(mydir + results_path + f'explainer_{explanation_type}.pkl', 'rb') as file:
#     explainer = pickle.load(file)
#
# # Carregar o objeto explanation do arquivo
# with open(mydir + results_path + f'explanation_{explanation_type}.pkl', 'rb') as file:
#     explanation = pickle.load(file)
