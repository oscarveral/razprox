"""Script para pintar un mapa de colores a partir de los resultados de Zonify.
"""

import argparse
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
from shapely.geometry import Point
from pathlib import Path
import time
from scipy.interpolate import griddata

import pandas as pd

if __name__ == "__main__":
    tic = time.time()

    # Parseamos los argumentos de entrada y salida: input_csv y output folder
    parser = argparse.ArgumentParser(
        prog="Color Map from FIS-Zonify",
        description="Generar un mapa de colores a partir de los resultados de FIS-Zonify."
    )
    parser.add_argument('input_csv')
    parser.add_argument('output_folder')
    parser.add_argument('-m', '--mask_shapefile', type=str, default=None,
                        help="Fichero shapefile para usar como máscara del mapa de colores.")
    args = parser.parse_args()
    input_csv = args.input_csv
    output_folder = Path(args.output_folder) / "color_map"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Cargamos los datos de entrada desde el fichero CSV
    print(f"Cargando datos de entrada desde {input_csv}")
    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"El fichero de entrada {input_csv} no existe.")
    data = pd.read_csv(input_csv, sep=',')
    print(f"Datos de entrada cargados. Número de filas: {len(data)}. Columnas: {data.columns.tolist()}")

    # Extraer los datos necesarios: Longitud, Latitud, r, g, b
    lon_min = data["Longitud"].min()
    lon_max = data["Longitud"].max()
    lat_min = data["Latitud"].min()
    lat_max = data["Latitud"].max()
    resolutionx = 0.01  # Grados
    resolutiony = 0.01  # Grados
    points = data[['Longitud', 'Latitud']].values
    colors = data[['r', 'g', 'b']].values.astype(float) / 255.0

    # Interpolar los colores en una cuadrícula regular
    lon_grid = np.arange(lon_min, lon_max + resolutionx, resolutionx)
    lat_grid = np.arange(lat_min, lat_max + resolutiony, resolutiony)
    LAT, LON = np.meshgrid(lat_grid, lon_grid)
    print("Interpolando colores para el mapa.")
    grid_coords = (LON.flatten(), LAT.flatten())
    R = griddata(points, colors[:, 0], grid_coords, method='nearest')
    G = griddata(points, colors[:, 1], grid_coords, method='nearest')
    B = griddata(points, colors[:, 2], grid_coords, method='nearest')
    R_2D = R.reshape(LON.shape).T
    G_2D = G.reshape(LON.shape).T
    B_2D = B.reshape(LON.shape).T

    # Aplicar máscara si se proporciona un shapefile
    if args.mask_shapefile is not None:
        print(f"Aplicando máscara súper optimizada desde shapefile {args.mask_shapefile}...")
        
        if not Path(args.mask_shapefile).is_file():
            raise FileNotFoundError(f"El fichero shapefile {args.mask_shapefile} no existe.")
            
        # 1. Cargar la geometría de la máscara (sin necesidad de union_all/unary_union)
        mask_gdf = gpd.read_file(args.mask_shapefile)
        
        # 2. Crear los puntos de la cuadrícula como un GeoDataFrame
        # a) Aplanar coordenadas del grid
        lons = LON.flatten()
        lats = LAT.flatten()
        
        # b) Crear una columna de geometría (Puntos) y un índice de píxel plano
        # Usamos el índice de píxel plano para mapear de vuelta a la matriz 2D
        flat_index = np.arange(lons.size)
        
        grid_points_gdf = gpd.GeoDataFrame(
            {'flat_index': flat_index},
            geometry=gpd.points_from_xy(lons, lats),
            crs=mask_gdf.crs # Asumimos que el CRS es el mismo
        )
        
        # 3. Realizar el Spatial Join (sjoin)
        # El sjoin usa el R-tree para encontrar qué puntos están *dentro* de los polígonos
        print("Realizando sjoin optimizado (consulta de pertenencia)...")
        
        # sjoin(left_df, right_df, predicate='within') encuentra puntos en left_df
        # que están dentro de polígonos en right_df.
        points_in_mask = grid_points_gdf.sjoin(
            mask_gdf[['geometry']], # Solo necesitamos la geometría de la máscara
            how='inner',            # Solo mantener los puntos que tienen coincidencia (están DENTRO)
            predicate='within'
        )
        
        # 4. Crear la máscara booleana a partir de los resultados
        # Crear un array plano de False (MAR/FUERA) del tamaño total del grid
        is_inside_mask_flat = np.full(lons.size, False, dtype=bool)
        
        # Marcar como True (TIERRA/DENTRO) todos los puntos que resultaron del sjoin
        is_inside_mask_flat[points_in_mask['flat_index']] = True
        
        # 5. Reestructurar y aplicar la máscara
        # La forma 2D de la máscara
        mask = is_inside_mask_flat.reshape(LON.shape).T

        # Aplicar el color de fondo (1.0 = Blanco; 0.0 = Negro) fuera de la máscara
        R_2D[~mask] = 1.0  
        G_2D[~mask] = 1.0
        B_2D[~mask] = 1.0

        print("Máscara aplicada correctamente.")


    image_array_float = np.dstack((R_2D, G_2D, B_2D))
    image_array_float_inverted = np.flipud(image_array_float)
    image_array_uint8 = (np.clip(image_array_float_inverted, 0, 1) * 255).astype(np.uint8)

    h, w, _ = image_array_uint8.shape
    print(f"Generando mapa de colores con resolución {w}x{h} píxeles...")
    img = Image.fromarray(image_array_uint8, 'RGB')
    img.save(output_folder / "color_map.png")


    # width_pixels = int((lon_max - lon_min) / resolutionx) + 1
    # height_pixels = int((lat_max - lat_min) / resolutiony) + 1

    # print(f"Generando mapa de colores con resolución {width_pixels}x{height_pixels} píxeles...")
    # data['col'] = np.floor((data['Longitud'] - lon_min) / resolutionx).astype(int)
    # data['row'] = np.floor((data['Latitud'] - lat_min) / resolutiony).astype(int)

    # image_array = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
    # image_array[data['row'], data['col']] = data[['r', 'g', 'b']].values
    # img = Image.fromarray(image_array, 'RGB')
    # img.save(output_folder / "color_map.png")

    

