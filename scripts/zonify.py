"""Script para evaluar y graficar el sistema de inferencia difusa FIS-Zonify.

Este script recibe como entrada un fichero CSV con columnas Longitud, Latitud, ABT, APP (mm) y PER.
Genera como salida un fichero CSV con las mismas columnas mas seis columnas adicionales 
Z1, Z2, Z3, r, g, b, siendo Z1, Z2 y Z3 las zonas de vida con mayor grado de pertenencia y r, g, b los colores RGB asociados, no
a la zona, sino a una ponderación de los valores de pertenencia de los diferentes conjuntos difusos.
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
FIS_ZONIFY_FILE = CONFIGS / "FIS-Zonify.json"

if __name__ == "__main__":
    tic = time.time()

    # Parseamos los argumentos de entrada y salida: input_csv y output folder
    parser = argparse.ArgumentParser(
        prog="FIS-Zonify",
        description="Evaluar y graficar el sistema de inferencia difusa FIS-Zonify."
    )
    parser.add_argument('input_csv')
    parser.add_argument('-of', '--output_folder', type=str, default=None,
                        help="Carpeta de salida para los resultados. Si no se especifica, se muestra por pantalla el gráfico y no se guarda ningún fichero.")
    parser.add_argument('-im', '--inference_mode', type=str, default='mandami',
                        help="Modo de inferencia difusa: 'mandami' o 'larsen'. Por defecto 'mandami'.")

    args = parser.parse_args()
    input_csv = args.input_csv
    output_folder = args.output_folder
    inference_mode = args.inference_mode

    # Cargamos los datos de entrada desde el fichero CSV
    log(f"Cargando datos de entrada desde {input_csv}")
    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"El fichero de entrada {input_csv} no existe.")
    data = pd.read_csv(input_csv, sep=',')
    log(f"Datos de entrada cargados. Número de filas: {len(data)}. Columnas: {data.columns.tolist()}")
    # Convertir las columnas a formato numérico, manejando comas como separadores decimales
    data["Longitud"] = pd.to_numeric(data["Longitud"], errors='coerce')
    data["Latitud"] = pd.to_numeric(data["Latitud"], errors='coerce')
    data["ABT"] = pd.to_numeric(data["ABT"], errors='coerce')
    data["APP"] = pd.to_numeric(data["APP"], errors='coerce')
    data["PER"] = pd.to_numeric(data["PER"], errors='coerce')

    # Cargar los datos de las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)
    log("Variables difusas cargadas correctamente.")

    # Cargar el sistema de inferencia difusa FIS-Zonify desde el archivo JSON
    fis = load_fis(FIS_ZONIFY_FILE, variables)
    log("Sistema de inferencia difusa cargado correctamente.")

    # Recorremos el dataframe evaluando FIS-Zonify para cada fila
    log("Evaluando FIS-Zonify para cada fila del dataset. Modo de inferencia: " + inference_mode)
    for index, row in data.iterrows():
        # Preparar la entrada para el sistema de inferencia
        try:
            abt_norm = math.log2(row["ABT"] / 0.75) if row["ABT"] > 0.375 else -1
            app_norm = math.log2(row["APP"] / 1000)
            per_norm = math.log2(row["PER"] / 2)
        except Exception as e:
            print(f"Error normalizando datos para la fila {index}: ABT={row['ABT']}, APP={row['APP']}, PER={row['PER']}")
            raise e
        input_data = {
            "ABT": abt_norm,
            "APP": app_norm,
            "PER": per_norm
        }
        # Evaluar el sistema de inferencia
        var, output = fis.eval(input_data, mode=inference_mode)
        try:
            color = var.defuzzify_color(output)
        except Exception as e:
            print(f"Error defuzzificando color para la fila {index+1} con datos de entrada: ABT={row['ABT']}, APP={row['APP']}, PER={row['PER']}")
            for fs_name, degree in output.items():
                print(f"    Fuzzy set '{fs_name}': degree of membership = {degree:.4f}")
            raise e

        data.at[index, "r"] = color[0]
        data.at[index, "g"] = color[1]
        data.at[index, "b"] = color[2]
        # Obtener las tres zonas con mayor grado de pertenencia desde el output, que es un diccionario[str,float]
        sorted_zones = sorted(output.items(), key=lambda item: item[1], reverse=True)
        data.at[index, "Z1"] = sorted_zones[0][0]
        data.at[index, "Z2"] = sorted_zones[1][0]
        data.at[index, "Z3"] = sorted_zones[2][0]
    log("Evaluación completada.")

    if output_folder is not None:
        log(f"Guardando resultados en carpeta {output_folder}")
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        output_csv = output_path / "zonify_results.csv"
        data.to_csv(output_csv, index=False)
        log(f"Resultados guardados en {output_csv}")

    # Ploteamos los resultados:
    latitudes = data["Latitud"].values
    longitudes = data["Longitud"].values
    colors = data[["r", "g", "b"]].values
    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(longitudes, latitudes, c=colors/255.0, marker='.', label="Puntos")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Zonificación climática desde FIS-Zonify")
    
    # Add legend on the side
    zonadevida = variables["ZonaDeVida"]
    from matplotlib.lines import Line2D
    colors = zonadevida.colors
    # Two columns legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=zone,
                              markerfacecolor=(color[0]/255.0, color[1]/255.0, color[2]/255.0), markersize=10)
                       for zone, color in colors.items()]
    ax.legend(
        handles=legend_elements, 
        title="Zonas de Vida", 
        #bbox_to_anchor=(0.5, -0.15), 
        loc='lower right', ncol=3,
        frameon=True
    )
    plt.tight_layout()


    if output_folder is None:
        plt.show()
    else:
        plt.savefig(output_path / "zonify_scatter_plot.png")

    # Creamos un fichero para alamacenar los metadatos de la ejecución
    if output_folder is not None:
        metadata_file = output_path / "zonify_execution_metadata.txt"
        with metadata_file.open('w') as f:
            f.write("Metadatos de la ejecución de FIS-Zonify\n")
            f.write(f"Input CSV: {input_csv}\n")
            f.write(f"Inference Mode: {inference_mode}\n")
            f.write(f"Execution Time: {time.time() - tic} seconds\n")





