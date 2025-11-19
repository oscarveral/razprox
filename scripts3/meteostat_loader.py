# Este script recibe como argumento un fichero shp y descarga datos meteorológicos de meteostat de los puntos 
# de tierra indicados en el shp, guardándolos en ficheros csv.

import argparse
import os
from pathlib import Path
from shapely.geometry import Point as ShpPoint
from meteostat import Point as MetPoint, Monthly, Stations
from datetime import datetime
import pandas as pd
import numpy as np
from shapely.geometry import box
import time
import geopandas as gpd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Meteostat Data Downloader",
        description="Descarga datos meteorológicos de meteostat para puntos de un fichero shp."
    )
    parser.add_argument('input_shp', help='Fichero shp con los puntos de tierra')
    parser.add_argument('output_dir', help='Directorio donde se guardarán los ficheros csv con los datos descargados')
    args = parser.parse_args()
    input_shp = args.input_shp
    output_dir = args.output_dir

    # Crear el directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- configuración: bbox aproximado para "Europa" (ajusta si quieres otro límite)
    lon_min, lon_max = -20, 65.0   # oeste, este
    lat_min, lat_max = 20.0, 80.0    # sur, norte
    step = 0.1                    # grados

    # --- 1) crear malla de puntos
    print("Creando malla de puntos...")
    lons = np.arange(lon_min, lon_max + 1e-9, step)
    lats = np.arange(lat_min, lat_max + 1e-9, step)
    xx, yy = np.meshgrid(lons, lats)
    pts = [ShpPoint(lon, lat) for lon, lat in zip(xx.ravel(), yy.ravel())]
    df = pd.DataFrame({'lon': xx.ravel(), 'lat': yy.ravel()})
    gdf_pts = gpd.GeoDataFrame(df, geometry=pts, crs="EPSG:4326")

    print("Cargando polígonos de tierra...")
    # --- 2) cargar polígonos de tierra (Natural Earth 'land' shapefile)
    land = gpd.read_file(input_shp).to_crs(epsg=4326)  # pon la ruta a tu shapefile
    # --- 3) recortar polígonos de tierra al bbox para acelerar (opcional)
    print("Recortando polígonos de tierra al bbox...")
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    land = land.clip(bbox)  # requiere geopandas >= 0.10; si no: land = land[land.intersects(bbox)]

    print("Realizando unión espacial para filtrar puntos sobre tierra...")
    # --- 4) spatial join: quedarse solo con puntos que intersectan tierra
    # (asegúrate que ambos GeoDataFrames usan EPSG:4326)
    joined = gpd.sjoin(gdf_pts, land, how="inner", predicate="intersects")
    # joined ahora contiene solo puntos sobre tierra

    # --- 5) guardar o usar
    print("Filtrando puntos únicos sobre tierra...")
    pairs = joined[['lat','lon']].drop_duplicates().sort_values(['lat','lon'])
    print(f"Número de puntos sobre tierra: {len(pairs)}")
    pairs.reset_index(drop=True, inplace=True)


    inicio = datetime(2010, 1, 1)
    fin = datetime(2024, 12, 31)

    stations_ids = []
    stations_locs = {}

    n_pairs = len(pairs)

    print("Buscando estaciones meteorológicas cercanas a los puntos de tierra...")
    for index, row in pairs.iterrows():
        stations = Stations()
        stations = stations.nearby(row['lat'], row['lon'])
        station = stations.fetch(1)
        if station.empty:
            continue
        stations_ids.append(station.index[0])
        stations_locs[station.index[0]] = (row['lat'], row['lon'])
        # Print progress every 5%
        if index % max(1, n_pairs // 20) == 0:
            progreso = (index / n_pairs) * 100
            print(f"Progreso en búsqueda de estaciones: {progreso:.1f}% ({index}/{n_pairs})")

    unique_stations_idx = list(set(stations_ids))

    print("Stations found:", len(unique_stations_idx))

    df = pd.DataFrame(columns=['StationID','Latitud', 'Longitud', 'ABT', 'APP', 'PER'])

    cont_missing = 0
    count = 0

    for station_id in unique_stations_idx:
        loc_data = Monthly(station_id, inicio, fin)
        # Descargar datos diarios desde 2010 hasta hoy
        loc_data = loc_data.fetch()
        if loc_data.empty:
            cont_missing += 1
            continue
        loc_data.drop(columns=['tavg', 'wspd', 'pres', 'tsun'], inplace=True, errors='ignore')
        # Si la tabla tiene algun NA, saltar este punto
        if loc_data.isnull().values.any():
            cont_missing += 1
            continue
        count += 1

        # 1. Calcular la temperatura media mensual
        loc_data['T_media'] = (loc_data['tmin'] + loc_data['tmax']) / 2

        # 2. Aplicar la función clip para obtener la Biotemperatura Mensual
        # clip(valor, limite_inferior, limite_superior)
        loc_data['BT_mensual'] = np.clip(loc_data['T_media'], 0, 30)

        abt_anual = loc_data.groupby(loc_data.index.year)['BT_mensual'].mean()
        prcp_anual = loc_data.groupby(loc_data.index.year)['prcp'].sum()

        abt = abt_anual.mean()
        prcp = prcp_anual.mean()
        per = abt / prcp * 58.93
        lat, lon = stations_locs[station_id]

        new_row = {
            'StationID': station_id,
            'Latitud': lat,
            'Longitud': lon,
            'ABT': abt,
            'APP': prcp,
            'PER': per
        }

        df_row = pd.DataFrame([new_row])
        df = pd.concat([df, df_row], ignore_index=True)

        # imprimir el progreso cada 5%
        if count % max(1, len(unique_stations_idx) // 20) == 0:
            progreso = (count / len(unique_stations_idx)) * 100
            print(f"Progreso: {progreso:.1f}% ({count}/{len(unique_stations_idx)})")

    print(f"Número de puntos sin datos completos: {cont_missing}")
        # Eliminar filas con datos faltantes
    df.dropna(inplace=True)

    print(df)
    # Guardar el DataFrame final en un archivo CSV
    output_csv = os.path.join(output_dir, 'meteostat_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"Datos guardados en {output_csv}")

    # Plotear los puntos con ABT y APP
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(80,60))
    land.plot(ax=ax, color="gray", edgecolor="black", label="Europa")
    fig.patch.set_facecolor('white')  # Establecer el fondo de la figura en blanco
    sc = ax.scatter(df['Longitud'], df['Latitud'], c=df['ABT'], cmap='viridis', s=50)
    plt.colorbar(sc, label='ABT (°C)', ax=ax)

    # Plot shapefile as background
    
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Puntos con ABT')
    plt.grid()
    plt.show()
    


    
 

    



    