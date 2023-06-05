import bz2
import pickle
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt
from torch_geometric_temporal import temporal_signal_split
from create_dataframe import DatasetLoader

warnings.filterwarnings("ignore")

lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCLSTM'
reps = 10

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
else:
    datasetLoader = DatasetLoader()
    dataset, dataset_standardized = datasetLoader.get_dataset(lags=lags, datasets=datasets, weight=weight, out=out)
    file = bz2.BZ2File(mydir + '/save/class/' + filename_dataset, 'wb')
    pickle.dump(datasetLoader, file)
    file.close()
    print('Dataset carregado e gerado arquivo pickle')

print("Separando dataset em teste e treinamento")

train_dataset, test_dataset = temporal_signal_split(dataset_standardized, train_ratio=0.8)

# Dias dos treino
days_train = [(pd.to_datetime("02/25/2020") + pd.DateOffset(days=x)).strftime('%Y.%m.%d') for x in
              range(train_dataset.snapshot_count * 14)]
# Dias dos testes
days_test = [(pd.to_datetime("02/25/2020") + pd.DateOffset(days=x)).strftime('%Y.%m.%d') for x in
             range(train_dataset.snapshot_count * 14,
                   (train_dataset.snapshot_count + test_dataset.snapshot_count) * 14 + 1)]

# Armazenando dados temporais de covid em treinamento e teste
train_data = np.zeros((train_dataset.snapshot_count * lags, (train_dataset.edge_index.max() + 1)))
for time, snapshot in enumerate(train_dataset):
    train_data[lags * time:lags * (time + 1)] = snapshot.x.T.cpu().detach().numpy()

test_data = np.zeros((test_dataset.snapshot_count * lags + 1, (test_dataset.edge_index.max() + 1)))
for time, snapshot in enumerate(test_dataset):
    test_data[lags * time:lags * (time + 1)] = snapshot.x.T.cpu().detach().numpy()
test_data[lags * (time + 1)] = snapshot.y.T.cpu().detach().numpy()


class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def create_inout_sequences(input_data, tw):
    return [(input_data[tw * i: tw * (i + 1)], input_data[tw * (i + 1)]) for i in range(((len(input_data) - 1) // tw))]


def sequence_lag(input_data, tw):
    return [input_data[tw * (i + 1)] for i in range(((len(input_data) - 1) // tw))]


train_data_no_norm = ((train_data * datasetLoader.std_stacked_dataset) + datasetLoader.mean_stacked_dataset)
test_data_no_norm = ((test_data * datasetLoader.std_stacked_dataset) + datasetLoader.mean_stacked_dataset)

# LSTM
pred_lstm = np.zeros((test_data_no_norm.shape[0] // 14, test_data_no_norm.shape[1]))
lstm_rmse_error_city = np.zeros((test_data_no_norm.shape[1]))
lstm_r2_city = np.zeros((test_data_no_norm.shape[1]))

# PROPHET
pred_prophet = np.zeros((test_data_no_norm.shape[0], test_data_no_norm.shape[1]))
prophet_rmse_error_city = np.zeros((test_data_no_norm.shape[1]))
prophet_r2_city = np.zeros((test_data_no_norm.shape[1]))

for city in range(train_data_no_norm.shape[1]):
    print('Cidade', city)
    print('GCLSTM')

    train_inout_seq = create_inout_sequences(torch.FloatTensor(train_data[:, city]).view(-1), lags)

    model = LSTM()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    test_inout_seq = create_inout_sequences(torch.FloatTensor(test_data[:, 0]).view(-1), lags)

    model.eval()

    test_outs = []
    for seq, labels in test_inout_seq:
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
            test_outs.append(model(seq).item())

    pred_lstm[:, city] = (
            (np.array(test_outs) * datasetLoader.std_stacked_dataset[0]) + datasetLoader.mean_stacked_dataset[0])
    # print(actual_predictions)

    test_targets = [
        ((np.float32(labels) * datasetLoader.std_stacked_dataset[0]) + datasetLoader.mean_stacked_dataset[0])
        for seq, labels in test_inout_seq]

    lstm_rmse_error_city[city] = mean_squared_error(test_targets, pred_lstm[:, city], squared=False)

    lstm_r2_city[city] = r2_score(test_targets, pred_lstm[:, city])

    # plt.figure()
    # plt.title('Número de casos no Tempo')
    # plt.ylabel('Número total de casos na Cidade')
    # plt.grid(True)
    # plt.autoscale(axis='x', tight=True)
    # plt.legend()

    print('Prophet')
    # dia 1: 2020-02-25 -> 1582599600
    train_data_pr = pd.DataFrame(train_data_no_norm[:, city], pd.to_datetime(days_train))
    train_data_pr = train_data_pr.reset_index()
    # dia 798: 2022-05-02
    test_data_pr = pd.DataFrame(test_data_no_norm[:, city], pd.to_datetime(days_test))
    test_data_pr = test_data_pr.reset_index()

    train_data_pr.columns = ['ds', 'y']
    test_data_pr.columns = ['ds', 'y']

    m = Prophet()
    m.fit(train_data_pr)
    future = m.make_future_dataframe(periods=test_data_no_norm[:, city].shape[0])
    prophet_pred = m.predict(future)

    pred_prophet[:, city] = prophet_pred['yhat'][-test_data_no_norm[:, 0].shape[0]:].values

    # plt.figure(figsize=(10, 8))
    # ax = sns.lineplot(x=[x for x in range(test_data_no_norm[:, city].shape[0])], y=test_data_no_norm[:, city])
    # sns.lineplot(x=[x for x in range(test_data_no_norm[:, city].shape[0])],
    #              y=prophet_pred['yhat'][-test_data_no_norm[:, city].shape[0]:].values)
    # plt.show()

    prophet_rmse_error_city[city] = mean_squared_error(test_data_no_norm[:, city], pred_prophet[:, city],
                                                       squared=False)

    prophet_r2_city[city] = r2_score(test_data_no_norm[:, 0],
                                     prophet_pred['yhat'][-test_data_no_norm[:, 0].shape[0]:].values)

#################

nomemun = []
for codemundv in datasetLoader.CODMUNDV:
    aux = datasetLoader._df_IBGE.loc[
        datasetLoader._df_IBGE.CODMUNDV_A == datasetLoader._label_encoder_CODMUNDV.transform([codemundv])[0]].NOMEMUN_A
    if aux.shape[0] == 0:
        aux = datasetLoader._df_IBGE.loc[
            datasetLoader._df_IBGE.CODMUNDV_B == datasetLoader._label_encoder_CODMUNDV.transform([codemundv])[
                0]].NOMEMUN_B
    nomemun.append(aux.iloc[0])

metrics = pd.DataFrame(
    {"NOMENUM": nomemun, "GCLSTM RMSE": lstm_rmse_error_city, "Prophet RMSE": prophet_rmse_error_city,
     "GCLSTM R2": lstm_r2_city, "Prophet R2": prophet_r2_city})

city = 0  # Manaus

plt.figure(figsize=(16, 9))
plt.plot_date(x=sequence_lag(test_data_pr.ds, lags), y=sequence_lag(test_data_no_norm[:, 0], lags), linestyle="-",
              label="test_data")
plt.plot_date(x=sequence_lag(test_data_pr.ds, lags), y=pred_lstm[:, 0], linestyle="--", label="pred_lstm")
plt.plot_date(x=sequence_lag(test_data_pr.ds, lags), y=sequence_lag(pred_prophet[:, 0], lags), linestyle=":",
              label="pred_prophet")
plt.legend(title=nomemun[city])
plt.show()

metrics.to_csv(path_or_buf=mydir + f'/results/{nameModel}/{datasets}/R2_RMSE_LSTM_Prophet.csv', sep=';',
               index=False)
np.save(mydir + f'/results/{nameModel}/{datasets}/pred_lstm.npy', pred_lstm)
np.save(mydir + f'/results/{nameModel}/{datasets}/pred_prophet.npy', pred_prophet)
