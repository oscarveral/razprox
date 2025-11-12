# This scripts generates a scatter plot.
# X axis goes from 0 to 90 (latitude)
# Y axis goes from 0 to 5000 (altitude)
# Points are colored based their predicted ABT value from FIS-Biotem.

import matplotlib.pyplot as plt
import math

from bioclas import load_variables, load_fis
from pathlib import Path

if __name__ == "__main__":
    RESOURCES_PATH = Path(__file__).parent.parent / "resources"
    VARIABLES_FILE = RESOURCES_PATH / "variables.json"
    FIS_BIOTEM_FILE = RESOURCES_PATH / "FIS-Biotem.json"
    DATOS_CUADRICULA_FILE = RESOURCES_PATH / "DATOS-CUADRICULA.csv"
    PENINSULA_SHP_FILE = RESOURCES_PATH / "ign" / "gadm41_ESP_0.shp"

    # Cargar las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)

    print("Cargando sistema de inferencia difusa para Biotem...")
    # Cargar el sistema de inferencia difusa para Biotem
    fis_biotem = load_fis(FIS_BIOTEM_FILE, variables)

    print("Sistemas de inferencia difusa cargados correctamente.")

    # Vamos a plotear los puntos del triangulo (0,0), (90,0), (0,5000)
    latitudes = []
    altitudes = []
    for lat in range(0, 90, 1):
        for alt in range(0, int(5000-500/9), 100):
            if alt <= (5000 - (5000/90)*lat):
                latitudes.append(lat)
                altitudes.append(alt)
    
    # Evaluar FIS-Biotem para cada punto y obtener los valores de ABT
    abt_values = []
    for lat, alt in zip(latitudes, altitudes):
        input_values = {
            "Latitud": lat,
            "Altitud": alt
        }
        
        var, output = fis_biotem.eval(input_values)
        abt = var.defuzzify(output)
        abt_values.append(abt)

    # Desnormaliazar ABT values
    abt_values = [0.75*math.exp2(abt) for abt in abt_values]

    # Dibujamos triangulo con los puntos
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(latitudes, altitudes, c=abt_values, cmap='viridis', marker='.', label="Puntos")
    ax.set_xlabel("Latitud")
    ax.set_ylabel("Altitud")
    ax.set_title("Predicted ABT values from FIS-Biotem")
    # Show colorbar legend
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('ABT Value')
    plt.grid(True)
    plt.show()
