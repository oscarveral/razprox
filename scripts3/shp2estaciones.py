import argparse
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point as ShpPoint
from meteostat import Monthly, Point as MetPoint, Stations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

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
    unique_station_locs = {sid: station_locs[sid] for sid in unique_station_ids}
    
    return unique_station_locs

tic = time.time()
# Parámetros de la malla y shapefile

lon_min, lon_max = -90.0, 90.0   # sur, norte
lat_min, lat_max = -180.0, 180.0    # oeste, este
step = 0.0625
input_shp = Path(__file__).parent.parent / "resources3" / "europe" / "Europe_merged.shp"
output_folder = Path(__file__).parent.parent / "resources3" / "estaciones"

lons = np.arange(lon_min, lon_max, step)
lats = np.arange(lat_min, lat_max, step)
xx, yy = np.meshgrid(lons, lats)

print("Creando malla de puntos sobre tierra...")

puntos = [(lon, lat) for lon, lat in zip(xx.ravel(), yy.ravel())]
print("Número total de puntos en la malla:", len(puntos))
puntos_sobre_tierra = puntos_tierra(input_shp, puntos)

print("Malla finalizada. Tiempo total de ejecución hasta este punto:", time.time() - tic, "segundos")

print(f"Número total de puntos en la malla: {len(puntos)}")
print(f"Número de puntos sobre tierra: {len(puntos_sobre_tierra)}")

print("Buscando estaciones meteorológicas cercanas a los puntos sobre tierra...")
estaciones = get_estaciones(puntos_sobre_tierra)

print("Búsqueda de estaciones finalizada. Tiempo total de ejecución:", time.time() - tic, "segundos")
print(f"Número total de estaciones encontradas: {len(estaciones)}")
print("Guardando estaciones en 'estaciones.csv'...")

# Create a DataFrame of the stations found (Dict {StationID: (lon, lat)}). Columns: StationID, Longitud, Latitud
df = pd.DataFrame(columns=['StationID','Longitud', 'Latitud'])
for station_id, (lon, lat) in estaciones.items():
    new_row = {'StationID': station_id, 'Longitud': lon, 'Latitud': lat}
    df_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_row], ignore_index=True)

df.to_csv(output_folder / "estaciones.csv", index=False)
print("Archivo guardado.")
print("Tiempo total de ejecución:", time.time() - tic, "segundos")

# Scatter plot de los puntos sobre tierra
plt.figure(figsize=(10, 6))
lon_tierra, lat_tierra = zip(*puntos_sobre_tierra)
plt.scatter(lon_tierra, lat_tierra, s=1, color='blue')
lon_estaciones, lat_estaciones = zip(*estaciones.values())
plt.scatter(lon_estaciones, lat_estaciones, s=3, color='red')
plt.title('Puntos sobre tierra')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.grid(True)
plt.savefig(output_folder / 'puntos_sobre_tierra.png', dpi=300)
plt.show()
