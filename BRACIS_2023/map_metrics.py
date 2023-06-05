import os
import igraph as ig
import numpy as np
from pathlib import Path

# https://plotly.com/python/mapbox-county-choropleth/
from urllib.request import urlopen
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def map_metrics(results_path, y_real, y_pred, datasetLoader, days_test):
    mydir = os.getcwd()

    # Todos os grafos criados
    files = os.listdir(mydir + '/networks/')

    # Ordenando
    files.sort(reverse=True)

    # Utilizamos o ultimo 'grafo_completo'
    file = files[-1]

    # Armazene o grafo
    g = ig.Graph.Read_GraphML(mydir + '/networks/' + file)

    # Atributos dos nós
    g.vertex_attributes()

    # Atributos das arestas
    g.edge_attributes()

    # Pastas das metricas
    relative_path = results_path  # + file[:-8]

    # O peso calculado sera de saida
    entradas = ['traffic_flow']  # 'in', 'out', 'time', 'cost'

    for entrada in entradas:
        g.vs["size"] = 12

        list_ibge_codes = list(datasetLoader._ibge_codes.keys())
        layout = []
        for i in range(g.vcount()):
            geocode = str(datasetLoader._label_encoder_CODMUNDV.inverse_transform([list_ibge_codes[i]])[
                              0])
            layout.append((g.vs.find(geocode=geocode)['LONG'], g.vs.find(geocode=geocode)["LATI"]))
        nr_vertices = g.vcount()

        position = {k: layout[k] for k in range(nr_vertices)}

        E = [e.tuple for e in g.es]  # list of edges

        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        for edge in E:
            Xe += [position[edge[0]][0], position[edge[1]][0]]
            Ye += [position[edge[0]][1], position[edge[1]][1]]

        labels = list(map(str, g.vs['NOMEMUN']))

        max = np.max(y_pred[0, :])

        # https://plotly.com/python/tree-plots/
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=Xe, y=Ye, mode='lines', line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='text', name='Route', ), )
        fig.add_trace(go.Scatter(x=Xn, y=Yn, mode='markers', name='City',
                                 marker=dict(
                                     size=15,  # define o tamanho dos pontos com base na magnitude
                                     color=y_pred[0, :] / max,  # define a cor dos pontos com base na magnitude
                                     colorscale="Viridis",  # define a escala de cores com base nas magnitudes
                                     opacity=0.8,
                                     colorbar=dict(title='Magnitude')  # adiciona uma legenda da escala de cores
                                 ),
                                 text=labels, hoverinfo='text', opacity=0.8, ))

        fig.update_layout(title='Number of COVID-19 cases in Brazil',
                          font_size=16,
                          showlegend=True,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          plot_bgcolor='rgb(248,248,248)'
                          )
        # fig.show()

        Path(mydir + f'/{relative_path}/maps/').mkdir(parents=True, exist_ok=True)
        fig.write_html(mydir + f'/{relative_path}/maps/' + file[:-8] + '_' + entrada + '_' + 'scatter.html')

        # https://github.com/tbrugz/geodata-br
        with urlopen(
                'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json') as response:
            counties = json.load(response)

        geocode = datasetLoader._label_encoder_CODMUNDV.inverse_transform(list_ibge_codes)
        nomemun = [
            g.vs.find(geocode=str(datasetLoader._label_encoder_CODMUNDV.inverse_transform([list_ibge_codes[x]])[0]))[
                'NOMEMUN'] for x in range(g.vcount())]
        df = pd.DataFrame(
            {'geocode': np.tile(geocode, y_pred.shape[0]), 'City name': np.tile(nomemun, y_pred.shape[0]),
             'Predicted number of cases': y_pred.flatten(), 'Real number of cases': y_real.flatten(),
             'day': np.repeat(days_test, y_pred.shape[1])})
        df['geocode'] = df['geocode'].astype('string')
        df['City name'] = df['City name'].astype('string')

        list_id = [counties["features"][_]['properties']['id'] for _ in range(counties["features"].__len__())]

        li1 = np.array(list_id)
        li2 = np.array(df.geocode.to_list())
        dif1 = np.setdiff1d(li1, li2, assume_unique=True)  # 184 retirar do li1
        dif2 = np.setdiff1d(li2, li1, assume_unique=True)  # 5 retirar do li2
        temp3 = np.concatenate((dif1, dif2))  # 189

        ref_index = 0
        count = 0
        for i in dif1:
            for j in range(ref_index, counties["features"].__len__()):
                if counties["features"][j]['properties']['id'] == i:
                    del counties["features"][j]  # deleta os elementos
                    count += 1
                    break
                ref_index += 1

        df = df.drop([df[df.geocode == _].index[0] for _ in dif2], )  # deleta os elementos
        df = df.reset_index(drop=True)

        for feature in counties['features']:
            feature['id'] = feature['properties']['id']

        fig = px.choropleth_mapbox(
            df,  # database
            color_continuous_scale="Viridis",
            locations="geocode",  # define the limits on the map/geography
            geojson=counties,  # shape information
            color="Predicted number of cases",  # defining the color of the scale through the database
            hover_name="City name",  # the information in the box
            hover_data=["City name", "geocode", "Predicted number of cases", "Real number of cases"],
            title="Number of COVID-19 cases in Brazil",  # title of the map
            animation_frame="day",  # creating the application based on the day
            mapbox_style="carto-positron",
            zoom=3, center={"lat": -15.0000, "lon": -55.0000},
            opacity=0.5,
            labels={'cases': 'Number of COVID-19 cases'}
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        # fig.show()
        fig.write_html(mydir + f'/{relative_path}/maps/' + file[:-8] + '_' + entrada + '_' + 'choropleth_mapbox.html')
