"""Script para generar un csv con puntos equiespaciados dentro del triangulo (0,0), (90,0), (5000,0)
en el espacio Latitud vs Altitud. Se almacenan dos columnas extra con -1 para Longitud y APP.
"""

from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    RESOURCES_PATH = Path(__file__).parent.parent / "resources"

    # Generar puntos equiespaciados en el triangulo
    latitudes = []
    altitudes = []
    longitudes = []
    apps = []
    step_lat = 1
    step_alt = int(5000/90)
    for lat in range(0, 91, step_lat):
        for alt in range(0, int(5000 - (5000/90)*lat) + 1, step_alt):
            latitudes.append(lat)
            altitudes.append(alt)
            longitudes.append(alt/step_alt)  # Valor fijo -1 para Longitud
            apps.append(1000)        # Valor fijo -1 para APP

    # Crear DataFrame y guardar a CSV
    data = pd.DataFrame({
        "Longitud": longitudes,
        "Latitud": latitudes,
        "Altitud": altitudes,
        "APP": apps
    })
    output_csv = RESOURCES_PATH / "triangle_points.csv"
    data.to_csv(output_csv, index=False)
    print(f"Puntos generados y guardados en {output_csv}")