"""Script para comparar los valores predichos por FIS-Biotem con valores calculados de forma directa con la regla empírica.
temperatura = temperatura_base - kilometros_snm * 6"""

import argparse
import math
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Biotem Error Evaluation",
        description="Comparar los valores predichos por FIS-Biotem con valores calculados de forma directa con la regla empírica."
    )
    parser.add_argument('input_csv')
    args = parser.parse_args()
    input_csv = args.input_csv

    # Cargamos los datos de entrada desde el fichero CSV
    data = pd.read_csv(input_csv, sep=',')
    # Convertir las columnas a formato numérico, manejando puntos como separadores decimales
    data["Latitud"] = pd.to_numeric(data["Latitud"], errors='coerce')
    data["Altitud"] = pd.to_numeric(data["Altitud"], errors='coerce')
    data["ABT"] = pd.to_numeric(data["ABT"], errors='coerce')

    # Obtenemos la temperatura base a nivel del mar en función de la latitud
    def get_base_temperature(lat):
        temp = 30 - 28.5 * (lat / 68)
        return max(temp, 0)


    data["Calculated_ABT"] = data.apply(lambda row: max(get_base_temperature(row["Latitud"]) - (row["Altitud"] / 1000) * 6, 0), axis=1)
    data["Error"] = data["ABT"] - data["Calculated_ABT"]
    data["Absolute_Error"] = data["Error"].abs()
    mae = data["Absolute_Error"].mean()
    max_error = data["Absolute_Error"].max()
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Max Absolute Error: {max_error}")
    plot_data = data.dropna(subset=["ABT", "Calculated_ABT"])
    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data["Calculated_ABT"], plot_data["ABT"], alpha=0.5)
    plt.plot([plot_data["Calculated_ABT"].min(), plot_data["Calculated_ABT"].max()],
             [plot_data["Calculated_ABT"].min(), plot_data["Calculated_ABT"].max()],
             color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel("Calculated ABT")
    plt.ylabel("FIS-Biotem Predicted ABT")
    plt.title("FIS-Biotem Predicted ABT vs Calculated ABT")
    plt.legend()
    plt.grid()
    plt.show()