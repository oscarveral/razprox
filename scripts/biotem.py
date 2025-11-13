"""Script para evaluar y graficar el sistema de inferencia difusa FIS-Biotem.

Este script recibe como entrada un fichero CSV con columnas Longitud, Latitud, Altitud y APP (mm).
Genera como salida un fichero CSV con las mismas columnas más una columna adicional ABT calculada a partir
del sistema de inferencia difusa FIS-Biotem.
"""

import argparse
import matplotlib.pyplot as plt
import math
from pathlib import Path
import time

import pandas as pd

from bioclas import load_variables, load_fis

def log(message: str):
    """Función simple para imprimir mensajes de log."""
    tic = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{tic}] {message}")

RESOURCES_PATH = Path(__file__).parent.parent / "resources"
VARIABLES_FILE = RESOURCES_PATH / "variables.json"
FIS_BIOTEM_FILE = RESOURCES_PATH / "FIS-Biotem.json"

if __name__ == "__main__":
    tic = time.time()

    # Parseamos los argumentos de entrada y salida: input_csv y output folder
    parser = argparse.ArgumentParser(
        prog="FIS-Biotem",
        description="Evaluar y graficar el sistema de inferencia difusa FIS-Biotem."
    )
    parser.add_argument('input_csv')
    parser.add_argument('output_folder')
    parser.add_argument('-imode', '--inference_mode', type=str, default='mandami',
                        help="Modo de inferencia difusa: 'mandami' o 'larsen'. Por defecto 'mandami'.")
    parser.add_argument('-dmethod', '--defuzzification_method', type=str, default='centroid',
                        help="Método de defuzzificación: 'centroid' o 'averageMax'. Por defecto 'centroid'.")
    args = parser.parse_args()
    input_csv = args.input_csv
    output_folder = args.output_folder
    inference_mode = args.inference_mode
    defuzzification_method = args.defuzzification_method

    
    # Cargamos los datos de entrada desde el fichero CSV
    log(f"Cargando datos de entrada desde {input_csv}")
    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"El fichero de entrada {input_csv} no existe.")
    data = pd.read_csv(input_csv, sep=',')
    log(f"Datos de entrada cargados. Número de filas: {len(data)}. Columnas: {data.columns.tolist()}")
    # Convertir las columnas a formato numérico, manejando comas como separadores decimales
    data["Longitud"] = pd.to_numeric(data["Longitud"], errors='coerce')
    data["Latitud"] = pd.to_numeric(data["Latitud"], errors='coerce')
    data["Altitud"] = pd.to_numeric(data["Altitud"], errors='coerce')
    data["APP"] = pd.to_numeric(data["APP"], errors='coerce')
    # Añadir columna ABT inicializada a NaN
    data["ABT"] = float('nan')

    # Cargar los datos de las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)

    log("Cargando sistema de inferencia difusa para Biotem.")
    # Cargar el sistema de inferencia difusa para Biotem
    fis_biotem = load_fis(FIS_BIOTEM_FILE, variables)

    log("Sistema de inferencia difusa cargados correctamente.")

    # Recorremos el dataframe evaluando FIS-Biotem para cada fila
    log("Evaluando FIS-Biotem para cada fila del dataset. Modo de inferencia: " + inference_mode)
    for index, row in data.iterrows():
        input_values = {
            "Latitud": row["Latitud"],
            "Altitud": row["Altitud"]
        }
        var, output = fis_biotem.eval(input_values, mode=inference_mode)
        abt = var.defuzzify(output, imode=inference_mode, method=defuzzification_method, step=0.01)
        # Desnormalizar ABT
        abt = 0.75*math.exp2(abt)
        data.at[index, "ABT"] = abt
    log("Evaluación completada.")
    log(f"Guardando resultados en carpeta {output_folder}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_path / "biotem_results.csv"
    data.to_csv(output_csv, index=False)
    log(f"Resultados guardados en {output_csv}")

    # Ploteamos los resultados: 
    latitudes = data["Latitud"].values
    altitudes = data["Altitud"].values
    abt_values = data["ABT"].values 
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(latitudes, altitudes, c=abt_values, cmap='viridis', marker='.', label="Puntos")
    ax.set_xlabel("Latitud")
    ax.set_ylabel("Altitud")
    ax.set_title("Predicted ABT values from FIS-Biotem")
    # Show colorbar legend
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('ABT Value')
    plt.savefig(output_path / "biotem_scatter_plot.png")

    # Creamos un fichero para alamacenar los metadatos de la ejecución
    metadata_file = output_path / "execution_metadata.txt"
    with metadata_file.open('w') as f:
        f.write("Metadatos de la ejecución de FIS-Biotem\n")
        f.write(f"Input CSV: {input_csv}\n")
        f.write(f"Output Folder: {output_folder}\n")
        f.write(f"Inference Mode: {inference_mode}\n")
        f.write(f"Defuzzification Method: {defuzzification_method}\n")
        f.write(f"Execution Time: {time.time() - tic} seconds\n")
    #plt.grid(True)
    #plt.show()