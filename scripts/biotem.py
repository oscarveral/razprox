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

CONFIGS = Path(__file__).parent.parent / "configs"
VARIABLES_FILE = CONFIGS / "variables.json"
FIS_BIOTEM_FILE = CONFIGS / "FIS-Biotem.json"

if __name__ == "__main__":
    tic = time.time()

    # Parseamos los argumentos de entrada y salida: input_csv y output folder
    parser = argparse.ArgumentParser(
        prog="FIS-Biotem",
        description="Evaluar y graficar el sistema de inferencia difusa FIS-Biotem."
    )
    parser.add_argument('input_csv')
    parser.add_argument('-of', '--output_folder', type=str, default=None,
                        help="Carpeta de salida para los resultados. Si no se especifica, se muestra por pantalla el gráfico y no se guarda ningún fichero.")
    parser.add_argument('-im', '--inference_mode', type=str, default='mandami',
                        help="Modo de inferencia difusa: 'mandami' o 'larsen'. Por defecto 'mandami'.")
    parser.add_argument('-dm', '--defuzzification_method', type=str, default='centroid',
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
    # Convertir las columnas a formato numérico, manejando puntos como separadores decimales
    data["Longitud"] = pd.to_numeric(data["Longitud"], errors='coerce')
    data["Latitud"] = pd.to_numeric(data["Latitud"], errors='coerce')
    data["Altitud"] = pd.to_numeric(data["Altitud"], errors='coerce')
    data["APP"] = pd.to_numeric(data["APP"], errors='coerce')
    # Añadir columnas ABT y PER inicializada a NaN
    data["ABT"] = float('nan')
    data["PER"] = float('nan')

    # Cargar los datos de las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)

    log("Cargando sistema de inferencia difusa para Biotem.")
    # Cargar el sistema de inferencia difusa para Biotem
    fis_biotem = load_fis(FIS_BIOTEM_FILE, variables)

    log("Sistema de inferencia difusa cargado correctamente.")

    # Recorremos el dataframe evaluando FIS-Biotem para cada fila
    log("Evaluando FIS-Biotem para cada fila del dataset. Modo de inferencia: " + inference_mode + ", Método de defuzzificación: " + defuzzification_method)
    for index, row in data.iterrows():
        input_values = {
            "Latitud": row["Latitud"],
            "Altitud": row["Altitud"]
        }
        var, output = fis_biotem.eval(input_values, mode=inference_mode)
        abt = var.defuzzify(output, imode=inference_mode, method=defuzzification_method, step=0.01)
        # Desnormalizar ABT
        abt = 0.75*math.exp2(abt)
        data.at[index, "ABT"] = round(abt, 2)
        # Calcular PER a partir de APP y ABT
        if row["APP"] >= 62.5:
            per = abt / row["APP"] * 58.93
        else:
            print(f"WARNING: APP demasiado bajo ({row['APP']}) en fila {index+1}. Se asigna APP=62.5.")
            data.at[index, "APP"] = 62.5
            per = abt / 62.5 * 58.93
        if (per < 0.125):
            print(f"WARNING: PER calculado demasiado bajo ({per}) en fila {index+1}. Se asigna PER=0.125.")
            per = 0.125
        data.at[index, "PER"] = round(per, 2)
    log("Evaluación completada.")

    if output_folder is not None:
        log(f"Guardando resultados en carpeta {output_folder}")
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        output_csv = output_path / "biotem_results.csv"
        data.to_csv(output_csv, index=False)
        log(f"Resultados guardados en {output_csv}")

    # Ploteamos los resultados: 
    latitudes = data["Latitud"].values
    longitudes = data["Longitud"].values
    abt_values = data["ABT"].values 
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(longitudes, latitudes, c=abt_values, cmap='viridis', marker='.', label="Puntos")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Predicted ABT values from FIS-Biotem")
    # Show colorbar legend
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('ABT Value')

    if output_folder is None:
        plt.show()
    else:
        plt.savefig(output_path / "biotem_scatter_plot.png")

    # Creamos un fichero para alamacenar los metadatos de la ejecución
    if output_folder is not None:
        metadata_file = output_path / "biotem_execution_metadata.txt"
        with metadata_file.open('w') as f:
            f.write("Metadatos de la ejecución de FIS-Biotem\n")
            f.write(f"Input CSV: {input_csv}\n")
            f.write(f"Output Folder: {output_folder}\n")
            f.write(f"Inference Mode: {inference_mode}\n")
            f.write(f"Defuzzification Method: {defuzzification_method}\n")
            f.write(f"Execution Time: {time.time() - tic} seconds\n")