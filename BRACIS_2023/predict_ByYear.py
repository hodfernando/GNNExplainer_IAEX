import os
import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import temporal_signal_split
from create_dataframe import DatasetLoader
from map_metrics import map_metrics
from models.gnn_lstm_model import LSTMGCN
from models.gnn_rnn_model import RecurrentGCN
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import bz2


class PredictByYear:

    def __init__(self, lags=14, datasets='2020-2022', weight='all', out='cases', nameModel='GCRN', reps=10,
                 train_ratio=0.8):
        self.lags = lags
        self.datasets = datasets
        self.weight = weight
        self.out = out
        self.nameModel = nameModel  # Qual modelo será treinado
        self.reps = reps  # Número de vezes que o teste será repetido
        self.train_ratio = train_ratio
        #
        self.datasetLoader = None
        self.model = None

    def predict(self):
        # Diretorios
        mydir = os.getcwd()
        Path(mydir + '/save/class/').mkdir(parents=True, exist_ok=True)
        files_class = os.listdir(mydir + '/save/class/')

        # Carregando Dataset
        filename_dataset = f'DATASET_lags_{self.lags}_datasets_{self.datasets}_weight_{self.weight}_out_{self.out}_'
        if filename_dataset in files_class:
            file = bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'rb')
            datasetLoader = pickle.load(file)
            dataset, dataset_standardized = datasetLoader.get_dataset(lags=self.lags, datasets=self.datasets,
                                                                      weight=self.weight, out=self.out)
            file.close()
            print('Dataset carregado pelo pickle data')
        else:
            self.datasetLoader = DatasetLoader()
            dataset, dataset_standardized = self.datasetLoader.get_dataset(lags=self.lags, datasets=self.datasets,
                                                                           weight=self.weight, out=self.out)
            file = bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'wb')
            pickle.dump(self.datasetLoader, file)
            file.close()
            print('Dataset carregado e gerado arquivo pickle')

        print("Separando dataset em teste e treinamento")

        train_dataset, test_dataset = temporal_signal_split(dataset_standardized, train_ratio=self.train_ratio)

        for rep in range(self.reps):
            # Carregando Model
            if self.nameModel == 'GCRN':
                self.model = RecurrentGCN(node_features=self.lags, filters=128)
            if self.nameModel == 'GCLSTM':
                self.model = LSTMGCN(node_features=self.lags, filters=128)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)  # adam aleatorio?

            self.model.train()
            print("Treinamento")
            time = 0
            for epoch in tqdm(range(100)):
                cost = 0
                for time, snapshot in enumerate(train_dataset):  # Shuffle
                    y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                    cost = torch.mean((y_hat - snapshot.y) ** 2)
                    cost.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            print("Fim do treinamento")

            print("Teste")

            self.model.eval()
            if rep == 0:
                y_real = np.zeros((self.reps, test_dataset.snapshot_count, test_dataset[0].y.shape[0]))
                y_pred = np.zeros((self.reps, test_dataset.snapshot_count, test_dataset[0].y.shape[0]))
                y_real_no_norm = np.zeros((self.reps, test_dataset.snapshot_count, test_dataset[0].y.shape[0]))
                y_pred_no_norm = np.zeros((self.reps, test_dataset.snapshot_count, test_dataset[0].y.shape[0]))
                rmse_all = np.zeros(self.reps)
                R2_all = np.zeros(self.reps)
                rmse_by_city = np.zeros((self.reps, test_dataset[0].y.shape[0]))
                R2_by_city = np.zeros((self.reps, test_dataset[0].y.shape[0]))

            time = 0
            for time, snapshot in enumerate(test_dataset):
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                y_real[rep, time] = snapshot.y.numpy()
                y_pred[rep, time] = y_hat.reshape(test_dataset[0].y.shape[0]).cpu().detach().numpy()  # cpu ou cuda

                # Revertendo a normalização
                y_real_no_norm[rep, time] = (
                        (y_real[rep, time] * datasetLoader.std_stacked_dataset) + datasetLoader.mean_stacked_dataset)
                y_pred_no_norm[rep, time] = (
                        (y_pred[rep, time] * datasetLoader.std_stacked_dataset) + datasetLoader.mean_stacked_dataset)

                R2_all[rep] = R2_all[rep] + r2_score(y_real_no_norm[rep, time], y_pred_no_norm[rep, time])
                rmse_all[rep] = rmse_all[rep] + mean_squared_error(y_real_no_norm[rep, time], y_pred_no_norm[rep, time],
                                                                   squared=False)

            rmse_all[rep] = np.divide(rmse_all[rep], (time + 1))
            R2_all[rep] = np.divide(R2_all[rep], (time + 1))

            for city in range(y_real.shape[2]):
                rmse_by_city[rep, city] = mean_squared_error(y_real_no_norm[rep, :, city], y_pred_no_norm[rep, :, city],
                                                             squared=False)
                R2_by_city[rep, city] = r2_score(y_real_no_norm[rep, :, city], y_pred_no_norm[rep, :, city])

            print("Fim do teste")

        #
        print(self.model)

        Path(mydir + f'/results/{self.nameModel}/{self.datasets}/').mkdir(parents=True, exist_ok=True)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/y_real_{self.datasets}.npy', y_real)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/y_pred_{self.datasets}.npy', y_pred)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/y_real_no_norm_{self.datasets}.npy', y_real_no_norm)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/y_pred_no_norm_{self.datasets}.npy', y_pred_no_norm)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/metric_R2_all_{self.datasets}.npy', R2_all)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/metric_RMSE_all_{self.datasets}.npy', rmse_all)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/metric_R2_by_city_{self.datasets}.npy',
                R2_by_city)
        np.save(mydir + f'/results/{self.nameModel}/{self.datasets}/metric_RMSE_by_city_{self.datasets}.npy',
                rmse_by_city)

        # print("RMSE normalizado: {:.4f}".format(rmse))
        print("RMSE mean: {:.4f}".format(np.mean(rmse_all)))
        print("RMSE max: {:.4f}".format(np.max(rmse_all)))
        print("RMSE min: {:.4f}".format(np.min(rmse_all)))
        print("RMSE std: {:.4f}".format(np.std(rmse_all)))
        print("R2 mean: {:.4f}".format(np.mean(R2_all)))
        print("R2 max: {:.4f}".format(np.max(R2_all)))
        print("R2 min: {:.4f}".format(np.min(R2_all)))
        print("R2 std: {:.4f}".format(np.std(R2_all)))

        # SP/SP - Sudeste - 3550308
        # Manaus/AM - Norte - 1302603
        # Fortaleza/CE - Nordeste - 2304400
        # Brasília/DF - Centro - 5300108
        # Florianópolis/SC - Sul - 4205407

        indexes = np.array(
            datasetLoader._label_encoder_CODMUNDV.transform([3550308, 1302603, 5300108]))

        nomes_municipios = ["" for x in range(datasetLoader._label_encoder_CODMUNDV.classes_.shape[0])]

        for i in range(indexes.shape[0]):
            aux = datasetLoader._df_IBGE.loc[datasetLoader._df_IBGE.CODMUNDV_A == indexes[i]].NOMEMUN_A
            if aux.shape[0] == 0:
                aux = datasetLoader._df_IBGE.loc[datasetLoader._df_IBGE.CODMUNDV_B == indexes[i]].NOMEMUN_B
            nomes_municipios[datasetLoader._ibge_codes.get(indexes[i])] = aux.iloc[0]

        # Dias dos testes
        days_test = [(pd.to_datetime("02/25/2020") + pd.DateOffset(days=(14 * (x + 1)))).strftime('%Y.%m.%d') for x
                     in range(train_dataset.snapshot_count, train_dataset.snapshot_count + test_dataset.snapshot_count)]

        y_real_mean_no_rep = np.mean(y_real_no_norm, axis=(0, 1))
        y_pred_mean_no_rep = np.mean(y_pred_no_norm, axis=(0, 1))

        # plotagem
        ind_x = [datasetLoader._ibge_codes.get(indexes[i]) for i in range(indexes.shape[0])]
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        ax.scatter(ind_x, y_real_mean_no_rep[ind_x], marker="x", s=300, c='red')
        ax.bar([x for x in range(y_pred_mean_no_rep.shape[0])], y_pred_mean_no_rep, color='blue', label="Y Predicted",
               width=30)
        ax.scatter([x for x in range(y_real_mean_no_rep.shape[0])], y_real_mean_no_rep, color='black', label="Y True")
        ax.legend(prop={'size': 16})
        plt.xticks([x for x in range(y_pred_mean_no_rep.shape[0])], nomes_municipios)
        plt.tick_params(axis='both', which='major', labelsize=16, )
        plt.xlabel('City Name', fontsize=16, )
        plt.ylabel('Number of Infected', fontsize=16, )
        plt.title('Number of infected by City', fontsize=16, )
        plt.grid(visible=None)
        plt.tight_layout()
        plt.show()

        fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=300)
        ax1.scatter(y_real_mean_no_rep, y_pred_mean_no_rep, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(y_pred_mean_no_rep), max(y_real_mean_no_rep))
        p2 = min(min(y_pred_mean_no_rep), min(y_real_mean_no_rep))
        ax1.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=16)
        plt.ylabel('Predictions', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16, )
        plt.axis('equal')
        plt.title('Predicted vs True', fontsize=16)
        plt.show()

        fig.savefig(mydir + f'/results/{self.nameModel}/{self.datasets}/InfecVsMun_{self.datasets}.png',
                    edgecolor='black', dpi=300, transparent=True)

        fig1.savefig(mydir + f'/results/{self.nameModel}/{self.datasets}/TruePredict_{self.datasets}.png',
                     edgecolor='black', dpi=300, transparent=True)

        map_metrics(results_path=f'/results/{self.nameModel}/{self.datasets}/', y_real=np.mean(y_real_no_norm, axis=0),
                    y_pred=np.mean(y_pred_no_norm, axis=0), datasetLoader=datasetLoader, days_test=days_test)
