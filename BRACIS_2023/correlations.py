import bz2
import os
import pickle
import numpy as np
import pandas as pd

from BRACIS_2023.map_percent_rmse import map_percent_rmse
from BRACIS_2023.map_rmse import map_rmse
import matplotlib.pyplot as plt

# Configurações base
lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCLSTM'

mydir = os.getcwd()
files_class = os.listdir(mydir + '/save/class/')

# Carregando Dataset
filename_dataset = f'DATASET_lags_{lags}_datasets_{datasets}_weight_{weight}_out_{out}_'
if filename_dataset in files_class:
    file = bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'rb')
    datasetLoader = pickle.load(file)
    dataset, dataset_standardized = datasetLoader.get_dataset(lags=lags, datasets=datasets, weight=weight, out=out)
    file.close()
    print('Dataset carregado pelo pickle data')

print("Separando dataset em teste e treinamento")

# GCLSTM
nameModel = 'GCLSTM'
rmse_gclstm = np.load(mydir + f'\\results\\{nameModel}\\{datasets}\\metric_RMSE_by_city_{datasets}.npy')

y_real_no_norm = np.load(mydir + f'\\results\\{nameModel}\\{datasets}\\y_real_no_norm_{datasets}.npy')
y_real_mean_test = np.mean(y_real_no_norm, axis=1)

percent_error_y_real_rmse_gclstm = np.divide(np.subtract(y_real_mean_test, rmse_gclstm), y_real_mean_test)

avg_percent_error_y_real_rmse_gclstm = percent_error_y_real_rmse_gclstm.mean(axis=0)

# Errou 90%
np.argwhere(avg_percent_error_y_real_rmse_gclstm < 0.1).shape  # 4
# Errou 50%
np.argwhere(avg_percent_error_y_real_rmse_gclstm < 0.5).shape  # 12
# Errou 30%
np.argwhere(avg_percent_error_y_real_rmse_gclstm < 0.7).shape  # 29
# Errou 20%
np.argwhere(avg_percent_error_y_real_rmse_gclstm < 0.8).shape  # 372

# GCRN
nameModel = 'GCRN'
rmse_gcrn = np.load(mydir + f'\\results\\{nameModel}\\{datasets}\\metric_RMSE_by_city_{datasets}.npy')

percent_error_y_real_rmse_gcrn = np.divide(np.subtract(y_real_mean_test, rmse_gcrn), y_real_mean_test)

avg_percent_error_y_real_rmse_gcrn = percent_error_y_real_rmse_gcrn.mean(axis=0)

# Errou 90%
np.argwhere(avg_percent_error_y_real_rmse_gcrn < 0.1).shape  # 4
# Errou 50%
np.argwhere(avg_percent_error_y_real_rmse_gcrn < 0.5).shape  # 11
# Errou 30%
np.argwhere(avg_percent_error_y_real_rmse_gcrn < 0.7).shape  # 23
# Errou 20%
np.argwhere(avg_percent_error_y_real_rmse_gcrn < 0.8).shape  # 232

# Prophet
rmse_prophet = np.load(mydir + f'\\results\\LSTM_Prophet\\pred_prophet.npy')

avg_percent_error_y_real_rmse_prophet = np.divide(np.subtract(y_real_mean_test.mean(axis=0), rmse_prophet.mean(axis=0)),
                                                  y_real_mean_test.mean(axis=0))

# LSTM
rmse_lstm = np.load(mydir + f'\\results\\LSTM_Prophet\\pred_lstm.npy')

avg_percent_error_y_real_rmse_lstm = np.divide(np.subtract(y_real_mean_test.mean(axis=0), rmse_lstm.mean(axis=0)),
                                               y_real_mean_test.mean(axis=0))

# Correlações
metrics = ['degree', 'betweenness', 'strength', 'betweenness_weight', 'closeness_weight', 'closeness']
data = pd.DataFrame()

saida = 'time'
for metric in metrics:
    df = pd.read_csv(mydir + f'/centralidades/grafo_completo/{saida}/' + '/' + metric + '.csv', delimiter=';', header=None, )
    df.columns = ['id', 'city_code', metric]
    if (data.columns.shape[0] == 0):
        data = df.copy()
    else:
        data[metric] = df[metric].copy()

data = data.fillna(value=0)  # preenchendo valores nan com 1

# Encontrar o índice correspondente aos códigos das cidades
indices = [np.where(datasetLoader.CODMUNDV == code)[0][0] for code in data.city_code]

# Preencher as colunas GCRN e GCLSTM com os valores correspondentes
data['GCRN'] = avg_percent_error_y_real_rmse_gcrn[indices]
data['GCLSTM'] = avg_percent_error_y_real_rmse_gclstm[indices]
data['Prophet'] = avg_percent_error_y_real_rmse_prophet[indices]
data['LSTM'] = avg_percent_error_y_real_rmse_lstm[indices]

# Selecionar as colunas desejadas
colunas = ['degree', 'betweenness', 'strength', 'betweenness_weight', 'closeness_weight', 'closeness', 'GCRN', 'GCLSTM',
           'Prophet', 'LSTM']
dados_selecionados = data[colunas]

# Calcular a correlação de Pearson
correlacao = dados_selecionados.corr()

# Imprimir a matriz de correlação
print(correlacao)
