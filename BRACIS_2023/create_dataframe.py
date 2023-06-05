import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class DatasetLoader(object):

    def __init__(self):
        # self._read_web_data()
        # Caracteristicas importantes
        self.NOMEMUN = []
        self._count_days = []  # Lista dos dias
        self.CODMUNDV = None  # Códigos de municipios
        self._targets = None  # Classes
        self._features = None  # Atributos
        self._targets_standardized = None  # Classes normalizados
        self._features_standardized = None  # Atributos normalizados
        self._standardized_stacked_dataset = None  # Stack normalizado
        self._std_stacked_dataset = None  # Desvio padrão do stack
        self._mean_stacked_dataset = None  # Valor da media do stack
        self._stacked_dataset = None  # Conversão do _dataset para stack
        self._dataset = None  # Conjunto dos dados agrupados a partir das caracteristica de entrada
        self._ibge_codes = None  # Dict dos codigos ibge
        self._label_encoder_CODMUNDV = None  # Encoder das comunidades
        self._df_IBGE = None  # Dataset do ibge
        self._df = None  # Dataset do conjunto de casos de covid
        self._edges = None  # Conjunto dos caminhos
        self._weights = None  # Conjunto dos pesos dos caminhos
        # Saida
        self._dataset_out = None  # Dataset que sera retornado
        self._dataset_out_standardized = None  # Dataset normalizado que sera retornado
        # Entradas
        self._lags = 14  # Define o intervalo temporal
        self._datasets = '2020'  # Define os datasets a serem utilizados
        self._weight = 'all'  # Define como sera calculado o peso
        self._out = 'newCases'  # Define qual caracteristica deseja prever

    def _read_web_data(self):
        # TODO :: Talvez seja interessante criar uma estrategia  de download dos dados via github do wcota e do ibge
        pass

    def generate_dataset(self):
        """Returning the IBGE + Cases Brazil Covid data.
        Args types:
            lags:
                - Número de dias do intervalo usado para previsão. Padrão 14 dias.
            datasets:
                - '2020', '2021', '2022' ou '2020-2021': Usa o dataset daquele ano especifico de casos de covid ou
                    um intervalo entre os anos.
            weight:
                - 'hidro': O peso da ligação entre dois municipios é o fluxo de pessoas apenas no meio de transporte hidrografico.
                - 'rod': O peso da ligação entre dois municipios é o fluxo de pessoas apenas no meio de transporte rodoviário.
                - 'veic': O peso da ligação entre dois municipios é o fluxo de pessoas apenas no meio de transporte não licenciado.
                - 'all': O peso da ligação entre dois municipios é o fluxo de pessoas da soma de todos os meios de transporte.
            out:
                - newCases: Saída é a previsão do número de novos casos no munipio.
                - cases: Saída é a previsão do número total de casos no munipio.
                - newDeaths: Saída é a previsão do número de novas mortes no munipio.
                - deaths: Saída é a previsão do número total de casos no munipio.
        Return types:
            * **dataset** *(Normalized StaticGraphTemporalSignal)*
            * **dataset** *(StaticGraphTemporalSignal)*
                - Graph IBGE + Cases Brazil Covid Dataset.
        """

        mydir = os.getcwd()
        print(f'lags={self._lags}, datasets={self._datasets}, weight={self._weight}, out={self._out}')

        print("Inicia get_dataset")
        # Armazena caminho do arquivo
        paths = [
            mydir + "/raw_data/cases-brazil-cities-time_2020.csv.gz",
            mydir + "/raw_data/cases-brazil-cities-time_2021.csv.gz",
            mydir + "/raw_data/cases-brazil-cities-time_2022.csv.gz",
        ]

        # Novo dataframe
        self._df = pd.DataFrame()

        # Setando intervalo
        intervalo_anos = list(map(int, [f[-1] for f in self._datasets.split('-')]))
        if intervalo_anos.__len__() == 1:
            intervalo_anos.append(intervalo_anos[0])

        # Coletando os dados de covid
        for each in range(intervalo_anos[0], intervalo_anos[1] + 1):
            self._df = pd.concat([self._df, pd.read_csv(paths[each])], ignore_index=True, )

        ## Carregando o dataset
        self._df_IBGE = pd.read_excel(mydir + "/raw_data/dataset_transform_IBGE.xlsx", header=0)

        # Conferindo se os codigos existentes da partida batem com os do destino
        # Transformando dados dos codigos dos municipios (categóricos) em numéricos
        self._label_encoder_CODMUNDV = LabelEncoder()
        self.CODMUNDV = pd.unique(self._df_IBGE[['CODMUNDV_A', 'CODMUNDV_B']].values.ravel('K'))  # 5385
        for codemundv in self.CODMUNDV:
            aux = self._df_IBGE.loc[self._df_IBGE.CODMUNDV_A == codemundv].NOMEMUN_A
            if aux.shape[0] == 0:
                aux = self._df_IBGE.loc[self._df_IBGE.CODMUNDV_B == codemundv].NOMEMUN_B
            self.NOMEMUN.append(aux.iloc[0])
        self._label_encoder_CODMUNDV.fit(self.CODMUNDV)
        self._df_IBGE.CODMUNDV_A = self._label_encoder_CODMUNDV.transform(self._df_IBGE.CODMUNDV_A.values)
        self._df_IBGE.CODMUNDV_B = self._label_encoder_CODMUNDV.transform(self._df_IBGE.CODMUNDV_B.values)

        # Verificando se existem valores nulos
        self._df_IBGE.isnull().values.any()  # Sem valores nulos

        # No dataset dos dados de covid, existem localidades que não tem código que bate com o código do ibge
        self._edges = self._df_IBGE[['CODMUNDV_A', 'CODMUNDV_B']].to_numpy().T  # Array partida destino

        # Funções de calculo do peso
        def hidro(_df):
            return _df['VAR05'].to_numpy()

        def rod(_df):
            return _df['VAR06'].to_numpy()

        def veic(_df):
            return _df['VAR12'].to_numpy()

        def all(_df):
            return (_df['VAR05'] + _df['VAR06'] + _df['VAR12']).to_numpy()

        # Calcula os pesos baseado na entrada
        self._weights = eval(self._weight + '(self._df_IBGE)')

        # Use o encoder para criar um dict de conversão dos indices do ibge em indices de uma lista
        self._ibge_codes = {self._label_encoder_CODMUNDV.transform([self.CODMUNDV[i]])[0]: i for i in
                            range(self.CODMUNDV.shape[0])}

        def newCases(_case):
            return _case.newCases

        def cases(_case):
            return _case.totalCases

        def newDeaths(_case):
            return _case.newDeaths

        def deaths(_case):
            return _case.deaths

        # Prepara o dataset com base nas caracteristicas de entrada
        self._dataset = np.zeros((self._df.date.unique().shape[0], self.CODMUNDV.shape[0]))
        self._count_days = []
        index_day = 0
        for day in self._df.date.unique():
            self._count_days.append(day)
            if self._out != "newCases" and self._out != "newDeaths" and index_day != 0:
                self._dataset[index_day][:] = self._dataset[index_day - 1][:].copy()
            for index_case, case in self._df[self._df.date == day].iterrows():
                if case.ibgeID != 0 and (case.ibgeID in self._label_encoder_CODMUNDV.classes_) and (
                        self._label_encoder_CODMUNDV.transform([case.ibgeID])[0] in self._ibge_codes):
                    self._dataset[index_day][
                        self._ibge_codes[self._label_encoder_CODMUNDV.transform([case.ibgeID])[0]]] = eval(
                        self._out + '(case)')
            index_day += 1

        # Transformando os dados
        self._stacked_dataset = np.stack(self._dataset)

        # Separa em atributos (dados usado para a previsão) e as classes (o dado que se deseja prever)
        self._features = [self._stacked_dataset[self._lags * i: self._lags * (i + 1), :].T for i in
                          range(((len(self._dataset) - 1) // self._lags))]
        self._targets = [self._stacked_dataset[self._lags * (i + 1), :].T for i in
                         range(((len(self._dataset) - 1) // self._lags))]

        # Salva o dataset e retorna
        self._dataset_out = StaticGraphTemporalSignal(self._edges, self._weights, self._features, self._targets)

        # Normalizando (Z) os dados
        self._mean_stacked_dataset = np.mean(self._stacked_dataset, axis=0)
        self._std_stacked_dataset = np.std(self._stacked_dataset, axis=0)
        numerador = (self._stacked_dataset - self._mean_stacked_dataset)
        denominador = self._std_stacked_dataset
        self._standardized_stacked_dataset = np.zeros(numerador.shape)
        np.divide(numerador, denominador, out=self._standardized_stacked_dataset, where=denominador != 0, )

        # arranjo = np.arange(0, 1009)
        # lags = 14
        # arranjo_feature = [arranjo[lags * i: lags * (i + 1)] for i in range(((len(arranjo) - 1) // lags))]
        # arranjo_target = [arranjo[lags * (i + 1)] for i in range(((len(arranjo) - 1) // lags))]

        # Separa em atributos (dados usado para a previsão) e as classes (o dado que se deseja prever)
        self._features_standardized = [self._standardized_stacked_dataset[self._lags * i: self._lags * (i + 1), :].T for
                                       i in range(((len(self._dataset) - 1) // self._lags))]
        self._targets_standardized = [self._standardized_stacked_dataset[self._lags * (i + 1), :].T for i in
                                      range(((len(self._dataset) - 1) // self._lags))]

        # Salva o dataset e retorna
        self._dataset_out_standardized = StaticGraphTemporalSignal(self._edges, self._weights,
                                                                   self._features_standardized,
                                                                   self._targets_standardized)
        print("Fim get_dataset")

    def get_dataset(self, lags=14, datasets='all', weight='all', out='newCases') -> [StaticGraphTemporalSignal,
                                                                                     StaticGraphTemporalSignal]:
        self._lags = lags
        self._datasets = datasets
        self._weight = weight
        self._out = out
        if self._dataset_out is None or self._dataset_out_standardized is None:
            self.generate_dataset()
        return self._dataset_out, self._dataset_out_standardized

    @property
    def std_stacked_dataset(self):
        return self._std_stacked_dataset

    @property
    def mean_stacked_dataset(self):
        return self._mean_stacked_dataset
