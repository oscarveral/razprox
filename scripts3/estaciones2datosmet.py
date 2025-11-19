import pandas as pd
from meteostat import Monthly
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

def get_meteorological_data(estaciones: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inicio = pd.Timestamp(2015, 1, 1)
    fin = pd.Timestamp(2020, 12, 31)

    columnas = ['Date', 'StationID', 'Latitud', 'Longitud', 'tmax', 'tmin', 'prcp']

    filas_completas = []
    filas_incompletas = []

    cont_missing = 0
    cont_incomplete = 0
    count_complete = 0

    for i, (station_id, (lon, lat)) in enumerate(estaciones.items()):
        if (i % max(1, len(estaciones) // 20)) == 0:
            progreso = (i / len(estaciones)) * 100
            print(f"Progreso: {progreso:.1f}% ({i}/{len(estaciones)})")

        data = Monthly(station_id, inicio, fin).fetch()

        if data.empty:
            cont_missing += 1
            continue

        data = data.drop(columns=['tavg', 'wspd', 'pres', 'tsun'], errors='ignore')

        destino = filas_incompletas if data.isnull().values.any() else filas_completas

        if destino is filas_incompletas:
            cont_incomplete += 1
        else:
            count_complete += 1

        for date, row in data.iterrows():
            destino.append({
                'Date': date,
                'StationID': station_id,
                'Latitud': lat,
                'Longitud': lon,
                'tmax': row.get('tmax', pd.NA),
                'tmin': row.get('tmin', pd.NA),
                'prcp': row.get('prcp', pd.NA)
            })

    # Ahora concatenamos UNA sola vez → sin FutureWarnings
    df = pd.DataFrame(filas_completas, columns=columnas)
    df_inc = pd.DataFrame(filas_incompletas, columns=columnas)

    print(f"Número de estaciones inaccesibles: {cont_missing}")
    print(f"Número de estaciones con datos incompletos: {cont_incomplete}")
    print(f"Número total de estaciones completas: {count_complete}")

    return df, df_inc

input = Path(__file__).parent.parent / "resources3" / "estaciones" / "estaciones.csv"
outputpath = Path(__file__).parent.parent / "resources3" / "datosmet"
outputpath.mkdir(exist_ok=True)

estaciones_df = pd.read_csv(input)
estaciones = {}
for _, row in estaciones_df.iterrows():
    estaciones[row['StationID']] = (row['Longitud'], row['Latitud'])

tic = time.time()
print("Descargando datos meteorológicos de las estaciones...")
df_complete, df_incomplete = get_meteorological_data(estaciones)
print("Descarga finalizada. Tiempo total de ejecución:", time.time() - tic, "segundos")

print("Guardando datos meteorológicos...")
df_complete.to_csv(outputpath / "datosmet_completos.csv", index=False)
df_incomplete.to_csv(outputpath / "datosmet_incompletos.csv", index=False)
print("Archivos guardados.")

# Scatter plot de las estaciones con datos completos e incompletos
plt.figure(figsize=(10, 6))
plt.scatter(df_incomplete['Longitud'], df_incomplete['Latitud'], c='red', label='Datos Incompletos', s=3)
plt.scatter(df_complete['Longitud'], df_complete['Latitud'], c='blue', label='Datos Completos', s=3)
plt.title('Estaciones Meteorológicas - Datos Completos vs Incompletos')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend()
plt.savefig(outputpath / "estaciones_datos_completos_incompletos.png")
plt.show()

