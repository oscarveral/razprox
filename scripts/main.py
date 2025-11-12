import geopandas as gpd
import matplotlib.pyplot as plt
import math
from pathlib import Path

from bioclas import load_variables, load_fis, load_geogrid

if __name__ == "__main__":
    RESOURCES_PATH = Path(__file__).parent.parent / "resources"
    VARIABLES_FILE = RESOURCES_PATH / "variables.json"
    FIS_ZONIFY_FILE = RESOURCES_PATH / "FIS-Zonify.json"
    FIS_BIOTEM_FILE = RESOURCES_PATH / "FIS-Biotem.json"
    DATOS_CUADRICULA_FILE = RESOURCES_PATH / "DATOS-CUADRICULA.csv"
    PENINSULA_SHP_FILE = RESOURCES_PATH / "ign" / "gadm41_ESP_0.shp"

    # Cargar las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)

    # Cargar el sistema de inferencia difusa para Zonify
    print("Cargando sistema de inferencia difusa para Zonify...")
    fis_zonify = load_fis(FIS_ZONIFY_FILE, variables)

    print("Cargando sistema de inferencia difusa para Biotem...")
    # Cargar el sistema de inferencia difusa para Biotem
    fis_biotem = load_fis(FIS_BIOTEM_FILE, variables)

    print("Sistemas de inferencia difusa cargados correctamente.")

    # Cargar los datos de la cuadricula
    print("Cargando datos de la cuadricula...")
    geogrid_data = load_geogrid(DATOS_CUADRICULA_FILE)
    print(f"Datos de la cuadricula cargados: {len(geogrid_data)} registros.")

    longitudes = [record[0] for record in geogrid_data]
    latitudes = [record[1] for record in geogrid_data]
    altitudes = [record[2] for record in geogrid_data]
    apps = [record[3] for record in geogrid_data]

    # Clippear app a 62.5 a 16000 y transformar a escala logarítmica
    apps = [max(62.5, min(16000, app)) for app in apps]
    apps_norm = [math.log2(app/1000) for app in apps]

    # Evaluar FIS-Biotem para cada punto y obtener los valores de ABT
    abts_norm = []
    for lat, alt in zip(latitudes, altitudes):
        input_values = {
            "Latitud": lat,
            "Altitud": alt
        }
        
        var, output = fis_biotem.eval(input_values)
        
        abt = var.defuzzify(output)
        abts_norm.append(abt)

    # Desnormalizar valores de ABT
    abts = [0.75 * math.exp2(abt) for abt in abts_norm]
    
    # Calcular PER
    pers = [58.93 * abt / app for abt, app in zip(abts, apps)]
    pers_norm = [math.log2(per/2.0) for per in pers]

    # Inferir zonas climáticas con FIS-Zonify
    zonas = []
    for abt, app, per in zip(abts_norm, apps_norm, pers_norm):
        input_values = {
            "ABT": abt,
            "APP": app,
            "PER": per
        }
        
        var, output = fis_zonify.eval(input_values)
        color = var.defuzzify_color(output)
        zonas.append(color)

    # Dibujamos mapa de españa
    try:
        peninsula = gpd.read_file(PENINSULA_SHP_FILE)
        fig, ax = plt.subplots(figsize=(80,60))
        fig.patch.set_facecolor('white')  # Establecer el fondo de la figura en blanco
        peninsula.plot(ax=ax, color="gray", edgecolor="black", label="España")
        # Color en la lista zonas en rgb
        sc = ax.scatter(longitudes, latitudes, color=[(r/255, g/255, b/255) for r, g, b in zonas], marker='o')
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title("Coordenadas con Silueta de la Península Ibérica")
        # Show color legend for zones
        colors = variables["ZonaDeVida"].colors
        # Crear una lista de parches para la leyenda. Colocarla fuera del gráfico
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=(r/255, g/255, b/255), label=name) for name, (r, g, b) in colors.items()]
        ax.legend(handles=legend_patches, title="Zonas de Vida", loc='upper left', bbox_to_anchor=(1, 1))   
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"No se pudo cargar el shapefile de la silueta: {e}")

