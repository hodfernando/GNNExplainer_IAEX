import bz2
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch_geometric_temporal import temporal_signal_split
from map_metrics import map_metrics
from scipy.stats import ttest_rel

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

print("Separando dataset em teste e treinamento")

train_dataset_standardized, test_dataset_standardized = temporal_signal_split(dataset_standardized, train_ratio=0.8)
train_dataset_original, test_dataset_original = temporal_signal_split(dataset, train_ratio=0.8)

# Dias dos treino
days_train = [(pd.to_datetime("02/25/2020") + pd.DateOffset(days=x)).strftime('%Y.%m.%d') for x in
              range(train_dataset_standardized.snapshot_count * 14)]
# Dias dos testes
days_test = [(pd.to_datetime("02/25/2020") + pd.DateOffset(days=x)).strftime('%Y.%m.%d') for x in
             range(train_dataset_standardized.snapshot_count * 14,
                   (train_dataset_standardized.snapshot_count + test_dataset_standardized.snapshot_count) * 14 + 1)]

# Armazenando dados temporais de covid em treinamento e teste
train_data_std = np.zeros(
    (train_dataset_standardized.snapshot_count * lags, (train_dataset_standardized.edge_index.max() + 1)))
for time, snapshot in enumerate(train_dataset_standardized):
    train_data_std[lags * time:lags * (time + 1)] = snapshot.x.T.cpu().detach().numpy()

test_data_std = np.zeros(
    (test_dataset_standardized.snapshot_count * lags + 1, (test_dataset_standardized.edge_index.max() + 1)))
for time, snapshot in enumerate(test_dataset_standardized):
    test_data_std[lags * time:lags * (time + 1)] = snapshot.x.T.cpu().detach().numpy()
test_data_std[lags * (time + 1)] = snapshot.y.T.cpu().detach().numpy()

train_data_ori = np.zeros((train_dataset_original.snapshot_count * lags, (train_dataset_original.edge_index.max() + 1)))
for time, snapshot in enumerate(train_dataset_original):
    train_data_ori[lags * time:lags * (time + 1)] = snapshot.x.T.cpu().detach().numpy()

test_data_ori = np.zeros(
    (test_dataset_original.snapshot_count * lags + 1, (test_dataset_original.edge_index.max() + 1)))
for time, snapshot in enumerate(test_dataset_original):
    test_data_ori[lags * time:lags * (time + 1)] = snapshot.x.T.cpu().detach().numpy()
test_data_ori[lags * (time + 1)] = snapshot.y.T.cpu().detach().numpy()


def sequence_lag(input_data, tw):
    return [input_data[tw * (i + 1)] for i in range(((len(input_data) - 1) // tw))]


new_x = []
new_y = np.append(train_data_ori.sum(axis=1), test_data_ori.sum(axis=1))
scatter_x = []
scatter_y = []
position = 0
for date in days_train + days_test:
    year = date[:4]
    if date == '2020.02.25' or date == '2021.01.01' or date == '2022.01.01':
        new_x.append(year)
        scatter_x.append(position)
        scatter_y.append(new_y[position])
    else:
        new_x.append('')
    position += 1

fig = plt.figure(figsize=(16, 9))
plt.plot(new_y, linestyle="-")
plt.scatter(x=scatter_x, y=scatter_y, color='red', s=100)
plt.xticks([x for x in range(new_x.__len__())], new_x)
plt.tick_params(axis='both', which='major', labelsize=24, )
plt.xlabel('Years', fontsize=32, )
plt.ylabel('Number of Infected', fontsize=32, )
plt.title('Number of infected in Brazil by year', fontsize=32, )
plt.tight_layout()
plt.grid(visible=None)
plt.show()

fig.savefig(mydir + f'/results/brazil.png', dpi=300)
fig.savefig(mydir + '/results/brazil.eps', format='eps', dpi=1200, bbox_inches='tight')

map_metrics(results_path=f'/results/', y_real=np.ones((1, 5385)), y_pred=np.ones((1, 5385)),
            datasetLoader=datasetLoader, days_test=[days_test[0]])

# plot time series
fig = plt.figure(figsize=(16, 9))
plt.plot(np.random.rand(100) * 10, linestyle="-", lw=5)
plt.xlabel('Time', fontsize=64, )
plt.ylabel('F(t)', fontsize=64, )
plt.title('Time Series for node i', fontsize=64, )
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.tight_layout()
# plt.grid()
plt.show()
fig.savefig(mydir + '/results/timeseries.png', dpi=300)
fig.savefig(mydir + '/results/timeseries.eps', format='eps', dpi=1200, bbox_inches='tight')

# Rankeamento BMA

# Definição dos modelos e suas probabilidades a priori
models = ['GCRN', 'GCLSTM', 'GCLSTM', 'Prophet']
prior_probs = [0.25, 0.25, 0.25, 0.25]  # prior probabilities
evidences = [0.01, 0.01, 0.01, 0.01]  # likelihoods (evidences)

# Dados da tabela
rmse_mean = [3059.50, 3583.88, 13267.66, 482.95]
rmse_max = [3699.74, 4569.97, 963263.36, 52058.21]
rmse_min = [2108.77, 2847.56, 115.91, 1.49]
rmse_std = [500.39, 452.59, 34109.31, 1758.85]

# Pesos de cada característica
weights_mean = 0.4
weights_max = 0.1
weights_min = 0.1
weights_std = 0.4

# normalize prior probabilities and evidences
prior_probs = np.exp(prior_probs) / np.sum(np.exp(prior_probs))
evidences = np.exp(evidences) / np.sum(np.exp(evidences))

posterior_probs = []
for i in range(len(models)):
    numerator = prior_probs[i] * evidences[i]
    denominator = np.sum([prior_probs[j] * evidences[j] for j in range(len(models))])
    posterior_prob = np.divide(numerator, denominator, where=denominator != 0)
    posterior_probs.append(posterior_prob)

# Cálculo da média ponderada dos resultados
weighted_rmse = np.zeros(len(models))
for i in range(len(models)):
    weighted_rmse[i] = (posterior_probs[i] * weights_mean * rmse_mean[i] +
                        posterior_probs[i] * weights_max * rmse_max[i] +
                        posterior_probs[i] * weights_min * rmse_min[i] +
                        posterior_probs[i] * weights_std * rmse_std[i])
averaged_rmse = np.sum(weighted_rmse)

# Criação do dataframe com os resultados
df_results = pd.DataFrame({
    "Model": models,
    "Weighted RMSE": weighted_rmse
})

# Ordenação do dataframe pelo RMSE ponderado em ordem crescente
df_results = df_results.sort_values("Weighted RMSE")

# Criando um gráfico de barras com Seaborn
sns.set(style='whitegrid')
ax = sns.barplot(x='Model', y='Weighted RMSE', data=df_results)

# Adicionando o valor de RMSE acima de cada barra
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=16, color='gray', xytext=(0, 10),
                textcoords='offset points')

# Adição das legendas e título
ax.set(xlabel='Model', ylabel='Weighted RMSE', title='BMA Ranking')
sns.despine(left=True, bottom=True)

# Mostrando o gráfico
plt.show()
fig = ax.get_figure()
fig.savefig(mydir + '/results/ranking_bma.png', dpi=300, bbox_inches='tight')
fig.savefig(mydir + '/results/ranking_bma.eps', format='eps', dpi=1200, bbox_inches='tight')

# Rankeamento normal
# Definindo os modelos e os valores de RMSE
models = ['GCRN', 'GCLSTM', 'GCLSTM', 'Prophet']
rmse_mean = [3059.50, 3583.88, 13267.66, 482.95]
rmse_max = [3699.74, 4569.97, 963263.36, 52058.21]
rmse_min = [2108.77, 2847.56, 115.91, 1.49]
rmse_std = [500.39, 452.59, 34109.31, 1758.85]

# Calculando o ranking dos modelos
ranking_rmse = []
for rmse in rmse_mean:
    rank = sum([1 for i in rmse_mean if i < rmse]) + 1
    ranking_rmse.append(rank)

# Criando um dicionário com os dados
data = {'Model': models,
        'Ranking': ranking_rmse}

# Criando um DataFrame a partir do dicionário
df = pd.DataFrame(data)

# Ordenando o DataFrame pelo valor de ranking (menor valor primeiro)
df = df.sort_values('Ranking')

# Criando um gráfico de barras com Seaborn
sns.set(style='whitegrid')
ax = sns.barplot(x='Model', y='Ranking', data=df)

# Configurando o título do gráfico e dos eixos
ax.set(xlabel='Model', ylabel='Ranking', title='Normal Ranking')
sns.despine(left=True, bottom=True)

# Mostrando o gráfico
plt.show()
fig = ax.get_figure()
fig.savefig(mydir + '/results/ranking.png', dpi=300, bbox_inches='tight')
fig.savefig(mydir + '/results/ranking.eps', format='eps', dpi=1200, bbox_inches='tight')

#
indexes = np.array(datasetLoader._label_encoder_CODMUNDV.transform([3550308, 1302603, 5300108]))

nomes_municipios = ["" for x in range(datasetLoader._label_encoder_CODMUNDV.classes_.shape[0])]

for i in range(indexes.shape[0]):
    aux = datasetLoader._df_IBGE.loc[datasetLoader._df_IBGE.CODMUNDV_A == indexes[i]].NOMEMUN_A
    if aux.shape[0] == 0:
        aux = datasetLoader._df_IBGE.loc[datasetLoader._df_IBGE.CODMUNDV_B == indexes[i]].NOMEMUN_B
    nomes_municipios[datasetLoader._ibge_codes.get(indexes[i])] = aux.iloc[0]

y_real_no_norm = np.load(mydir + f'\\results\\{nameModel}\\{datasets}\\y_real_no_norm_{datasets}.npy')
y_pred_no_norm = np.load(mydir + f'\\results\\{nameModel}\\{datasets}\\y_pred_no_norm_{datasets}.npy')
# y_real_mean_no_rep = np.mean(y_real_no_norm, axis=(0, 1))
# y_pred_mean_no_rep = np.mean(y_pred_no_norm, axis=(0, 1))

# Dia 29 11 2022
y_real_mean_no_rep = np.mean(y_real_no_norm, axis=0)[-1]
y_pred_mean_no_rep = np.mean(y_pred_no_norm, axis=0)[-1]

# plotagem
ind_x = [datasetLoader._ibge_codes.get(indexes[i]) for i in range(indexes.shape[0])]
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.scatter(ind_x, y_real_mean_no_rep[ind_x], marker="x", s=300, c='red')
ax.bar([x for x in range(y_pred_mean_no_rep.shape[0])], y_pred_mean_no_rep, color='blue', label="Y Predicted", width=30)
ax.scatter([x for x in range(y_real_mean_no_rep.shape[0])], y_real_mean_no_rep, color='black', label="Y True")
ax.legend(prop={'size': 32})
plt.xticks([x for x in range(y_pred_mean_no_rep.shape[0])], nomes_municipios)
plt.tick_params(axis='both', which='major', labelsize=32, )
plt.xlabel('City Name', fontsize=32, )
plt.ylabel('Number of Infected', fontsize=32, )
plt.title('Number of infected by City', fontsize=32, )
plt.grid(visible=None)
plt.tight_layout()
plt.show()

fig.savefig(mydir + f'/results/{nameModel}/{datasets}/InfecVsMun_day_29_11_2022_{nameModel}.png', dpi=300)
fig.savefig(mydir + f'/results/{nameModel}/{datasets}/InfecVsMun_day_29_11_2022_{nameModel}.eps', format='eps',
            dpi=1200,
            bbox_inches='tight')

# Calcular o valor p
_, p_value = ttest_rel(y_pred_mean_no_rep, y_real_mean_no_rep)

# Imprimir o valor p
print("O valor p é:", "{:.1e}".format(p_value))

fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=300)
ax1.scatter(y_real_mean_no_rep, y_pred_mean_no_rep, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(y_pred_mean_no_rep), max(y_real_mean_no_rep))
p2 = min(min(y_pred_mean_no_rep), min(y_real_mean_no_rep))
ax1.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=32)
plt.ylabel('Predictions', fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=16, )
plt.axis('equal')
plt.title('Predicted vs True', fontsize=32)
# plt.grid(visible=None)
ax1.annotate('p-value = {:.1e}'.format(p_value), xy=(p2, 0.1*p1), fontsize=28,)
plt.tight_layout()
plt.show()

fig1.savefig(mydir + f'/results/{nameModel}/{datasets}/TruePredict_day_29_11_2022_{nameModel}.png', dpi=300, )
fig1.savefig(mydir + f'/results/{nameModel}/{datasets}/TruePredict_day_29_11_2022_{nameModel}.eps', format='eps',
             dpi=1200, bbox_inches='tight')

# plt.figure(figsize=(16, 9))
# plt.plot_date(x=sequence_lag(test_data_pr.ds, lags), y=sequence_lag(test_data_no_norm[:, 0], lags), linestyle="-",
#               label="test_data")
# plt.plot_date(x=sequence_lag(test_data_pr.ds, lags), y=pred_lstm[:, 0], linestyle="--", label="pred_lstm")
# plt.plot_date(x=sequence_lag(test_data_pr.ds, lags), y=sequence_lag(pred_prophet[:, 0], lags), linestyle=":",
#               label="pred_prophet")
# plt.legend(title=nomemun[city])
# plt.show()
