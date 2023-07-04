import os
import igraph as ig
import numpy as np
from pathlib import Path

# https://plotly.com/python/mapbox-county-choropleth/
from urllib.request import urlopen
import json
import pandas as pd
import plotly.express as px


def map_percent_rmse(results_path, avg, datasetLoader, avg_percent_error, threshold):
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
    Path(mydir + f'/{relative_path}/maps/').mkdir(parents=True, exist_ok=True)

    # O peso calculado sera de saida
    # entradas = ['traffic_flow']  # 'in', 'out', 'time', 'cost'
    list_ibge_codes = list(datasetLoader._ibge_codes.keys())

    # for entrada in entradas:
    # https://github.com/tbrugz/geodata-br
    with urlopen(
            'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json') as response:
        counties = json.load(response)

    geocode = datasetLoader._label_encoder_CODMUNDV.inverse_transform(list_ibge_codes)
    nomemun = [
        g.vs.find(geocode=str(datasetLoader._label_encoder_CODMUNDV.inverse_transform([list_ibge_codes[x]])[0]))[
            'NOMEMUN'] for x in range(g.vcount())]
    df = pd.DataFrame({'geocode': geocode, 'City name': nomemun, 'Percent RMSE': avg.flatten()})
    df['geocode'] = df['geocode'].astype('string')
    df['City name'] = df['City name'].astype('string')

    list_id = [counties["features"][_]['properties']['id'] for _ in range(counties["features"].__len__())]

    li1 = np.array(list_id)
    li2 = np.array(df.geocode.to_list())
    li2_threshold = li2[np.argwhere(avg_percent_error < threshold)].flatten()
    dif1 = np.setdiff1d(li1, li2, assume_unique=True)  # 184 retirar do li1
    dif2 = np.setdiff1d(li2, li1, assume_unique=True)  # 5 retirar do li2
    # temp3 = np.concatenate((dif1, dif2))  # 189

    concatenated_array = np.concatenate((dif1, li2_threshold))
    unique_values = np.unique(concatenated_array)

    ref_index = 0
    count = 0
    for i in unique_values:
        for j in range(ref_index, counties["features"].__len__()):
            if counties["features"][j]['properties']['id'] == i:
                del counties["features"][j]  # deleta os elementos
                count += 1
                break
            ref_index += 1

    concatenated_array = np.concatenate((dif2, li2_threshold))
    unique_values = np.unique(concatenated_array)

    df = df.drop([df[df.geocode == _].index[0] for _ in unique_values], )  # deleta os elementos
    df = df.reset_index(drop=True)

    for feature in counties['features']:
        feature['id'] = feature['properties']['id']

    fig = px.choropleth_mapbox(
        df,  # database
        color_continuous_scale="Viridis",
        locations="geocode",  # define the limits on the map/geography
        geojson=counties,  # shape information
        color="Percent RMSE",  # defining the color of the scale through the database
        hover_name="City name",  # the information in the box
        hover_data=["City name", "geocode", "Percent RMSE"],
        title="RMSE COVID-19 cases in Brazil",  # title of the map
        mapbox_style="carto-positron",
        zoom=3, center={"lat": -15.0000, "lon": -55.0000},
        opacity=0.5,
        labels={'Percent RMSE': 'Percent RMSE COVID-19 cases'}
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    # fig.show()
    fig.write_html(
        mydir + f'/{relative_path}/maps/' + file[:-8] + '_' + 'choropleth_mapbox_percent_RMSE.html')
