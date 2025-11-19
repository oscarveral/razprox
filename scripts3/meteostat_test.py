import argparse
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point as ShpPoint
from meteostat import Monthly, Point as MetPoint, Stations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def puntos_tierra(shapefile: Path, puntos: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Dada una lista de puntos (lon, lat), devuelve solo los que caen sobre tierra
    según el shapefile proporcionado.
    """
    # Cargar el shapefile
    mapa_tierra = gpd.read_file(shapefile).to_crs(epsg=4326)

    # Convertir la lista de tuplas (lon, lat) a objetos Point (lon, lat)
    puntos_shapely = [ShpPoint(lon, lat) for lon, lat in puntos]

    # Crear un GeoDataFrame de los puntos
    gdf_puntos = gpd.GeoDataFrame(
        [f'Punto_{i+1}' for i in range(len(puntos_shapely))],
        geometry=puntos_shapely,
        crs="EPSG:4326"
    )

    # Realizar el spatial join
    puntos_en_tierra_gdf = gpd.sjoin(
        left_df=gdf_puntos,
        right_df=mapa_tierra,
        how='inner',
        predicate='within'
    )

    # Obtener los puntos originales (lon, lat) que están en tierra
    puntos_en_tierra_originales = [
        (lon, lat) for lon, lat in [(p.x, p.y) for p in puntos_en_tierra_gdf.geometry]
    ]

    return puntos_en_tierra_originales

def get_estaciones(puntos: list[tuple[float, float]]) -> list[MetPoint]:
    """Dada una lista de puntos (lon, lat), devuelve las estaciones meteorológicas
    más cercanas a cada punto usando la librería meteostat.
    """
    station_ids = []
    station_locs = {}

    print("Buscando estaciones meteorológicas cercanas a los puntos de tierra...")
    for i, (lon, lat) in enumerate(puntos):
        if (i % max(1, len(puntos) // 20)) == 0:
            progreso = (len(station_ids) / len(puntos)) * 100
            print(f"Progreso en búsqueda de estaciones: {progreso:.1f}% ({len(station_ids)}/{len(puntos)})")
        estaciones_cercanas = Stations().nearby(lat, lon).fetch(1)
        if estaciones_cercanas.empty:
            continue
        station_id = estaciones_cercanas.index[0]
        station_ids.append(station_id)
        station_locs[station_id] = (lon, lat)
    unique_station_ids = list(set(station_ids))
    print("Número de estaciones únicas encontradas:", len(unique_station_ids))
    unique_station_locs = {sid: station_locs[sid] for sid in unique_station_ids}
    
    return unique_station_locs

def get_meteorological_data(estaciones: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Dada una lista de estaciones meteorológicas (ID y ubicación), descarga
    los datos meteorológicos mensuales y devuelve un DataFrame con los datos
    relevantes.
    """
    inicio = pd.Timestamp(2010, 1, 1)
    fin = pd.Timestamp(2020, 12, 31)

    df = pd.DataFrame(columns=['StationID','Latitud', 'Longitud', 'ABT', 'APP', 'PER'])

    cont_missing = 0
    cont_incomplete = 0
    count = 0

    for i, (station_id, (lon, lat)) in enumerate(estaciones.items()):
        data = Monthly(station_id, inicio, fin)
        data = data.fetch()
        if data.empty:
            cont_missing += 1
            continue
        #print(data.head())
        
        data.drop(columns=['tavg', 'wspd', 'pres', 'tsun'], inplace=True, errors='ignore')
        # Si la tabla tiene algun NA, saltar este punto
        if data.isnull().values.any():
            cont_incomplete += 1
            continue

        data['T_media'] = (data['tmin'] + data['tmax']) / 2
        data['BT_mensual'] = np.clip(data['T_media'], 0, 30)
        abt_anual = data.groupby(data.index.year)['BT_mensual'].mean()
        prcp_anual = data.groupby(data.index.year)['prcp'].sum()
        abt = abt_anual.mean()
        app = prcp_anual.mean()
        per = abt / app * 58*93 if app > 0 else 32

        new_row = {
            'StationID': station_id,
            'Latitud': lat,
            'Longitud': lon,
            'ABT': round(abt,3),
            'APP': round(app,3),
            'PER': round(per,3)
        }

        count += 1

        df_row = pd.DataFrame([new_row])
        df = pd.concat([df, df_row], ignore_index=True)

        if (i % max(1, len(estaciones) // 20)) == 0:
            progreso = (i / len(estaciones)) * 100
            print(f"Progreso en descarga de datos meteorológicos: {progreso:.1f}% ({i}/{len(estaciones)})")

    print(f"Número de estaciones sin datos disponibles: {cont_missing}")
    print(f"Número de estaciones con datos incompletos: {cont_incomplete}")
    print(f"Número total de estaciones con datos descargados: {count}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Meteostat Land Points Filter",
        description=""
    )
    parser.add_argument('input_shp', help='Fichero shp con los polígonos de tierra')
    parser.add_argument('-s', "--step", type=float, default=1, help='Paso en grados para la malla de puntos (default: 0.1)')
    args = parser.parse_args()
    input_shp = Path(args.input_shp)
    step = args.step

    # Crear malla de puntos
    lon_min, lon_max = -24, 45.0   # oeste, este
    lat_min, lat_max = 3.0, 72.0    # sur, norte

    lons = np.arange(lon_min, lon_max + 1e-9, step)
    lats = np.arange(lat_min, lat_max + 1e-9, step)
    xx, yy = np.meshgrid(lons, lats)
    puntos = [(lon, lat) for lon, lat in zip(xx.ravel(), yy.ravel())]
    print(f"Número total de puntos en la malla: {len(puntos)}")
    puntos_sobre_tierra = puntos_tierra(input_shp, puntos)
    print(f"Número de puntos sobre tierra: {len(puntos_sobre_tierra)}")

    estaciones = get_estaciones(puntos_sobre_tierra)

    df = get_meteorological_data(estaciones)

    print("Datos meteorológicos descargados para las estaciones cercanas a los puntos sobre tierra:")
    print(df.head())
    df.to_csv("meteostat_land_points_data.csv", index=False)

    # Plotear los puntos y las estaciones
    plt.figure(figsize=(10, 8))
    plt.scatter(*zip(*puntos_sobre_tierra), s=1, color='blue', label='Puntos sobre tierra')
    plt.scatter(*zip(*estaciones.values()), s=10, color='red', label='Estaciones meteorológicas')
    # Plotear en morado los puntos que tienen datos meteorológicos
    estaciones_con_datos = [(lon, lat) for sid, (lon, lat) in estaciones.items() if sid in df['StationID'].values]
    plt.scatter(*zip(*estaciones_con_datos), s=20, color='purple', label='Estaciones con datos meteorológicos')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Puntos sobre tierra y estaciones meteorológicas cercanas')
    plt.legend()
    plt.show()



