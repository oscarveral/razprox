from pathlib import Path
import pandas as pd
import numpy as np


RESOURCES = Path(__file__).parent.parent / "resources"
ESTACIONES_FILE = RESOURCES / "LISTA-ESTACIONES.csv"
DATOS_CLIMATICOS_FOLDER = RESOURCES / "datos_climaticos"
OUTPUT_FILE = RESOURCES / "estaciones_procesadas.csv"

if __name__ == "__main__":
    # Cargar la lista de estaciones desde el fichero CSV
    estaciones_df = pd.read_csv(ESTACIONES_FILE, sep=',')
     # Latitud y Longitud están tienen coma como separador decimal, convertirlas a punto
    estaciones_df["Latitud"] = pd.to_numeric(estaciones_df["Latitud"].str.replace(',', '.'), errors='coerce')
    estaciones_df["Longitud"] = pd.to_numeric(estaciones_df["Longitud"].str.replace(',', '.'), errors='coerce')
    # Eliminamos Zona UTM, UTM X y UTM Y
    estaciones_df = estaciones_df.drop(columns=["Zona UTM", "UTM X", " UTM Y"])
    print(f"Número de estaciones cargadas: {len(estaciones_df)}")
    print("Primeras 5 estaciones:")
    print(estaciones_df.head())


    # Recorrer la carperta DATOS_CLIMATICOS_FOLDER abriendo cada fichero CSV
    for datos_file in DATOS_CLIMATICOS_FOLDER.glob("*.csv"):
        # El nombre del fichero empieza por Identificador_
        identificador = datos_file.stem.split('_')[0]
        # Buscar la estación correspondiente en estaciones_df
        estacion = estaciones_df[estaciones_df['Identificador'] == identificador]
        if estacion.empty:
            print(f"Estación con identificador {identificador} no encontrada en la lista de estaciones.")
            continue

        # Cargar los datos climáticos desde el fichero CSV
        # Hay caracteres utf-8 especiales, usar encoding='utf-8-sig'
        print(f"Cargando datos climáticos para la estación {identificador}")
        datos_df = pd.read_csv(datos_file, sep=';', encoding='utf-16-le', encoding_errors='ignore')
        datos_df['Precipitación (mm)'] = pd.to_numeric(datos_df['Precipitación (mm)'].str.replace(',', '.'), errors='coerce')
        datos_df['Temp Max (ºC)'] = pd.to_numeric(datos_df['Temp Max (ºC)'].str.replace(',', '.'), errors='coerce')
        datos_df['Temp Mínima (ºC)'] = pd.to_numeric(datos_df['Temp Mínima (ºC)'].str.replace(',', '.'), errors='coerce')

        anos = 0
        abt = 0.0
        app = 0.0
        for i in range(2018, 2024):
            datos_ano = datos_df[datos_df['Año'] == i]
            if datos_ano.empty:
                print(f"  No hay datos para el año {i} en la estación {identificador}.")
                continue
            app+= datos_ano['Precipitación (mm)'].sum()
            
            tmax = datos_ano['Temp Max (ºC)'].clip(lower=0.0).to_list()
            tmin = datos_ano['Temp Mínima (ºC)'].clip(lower=0.0).to_list()
            abt += sum([(tmax[m] + tmin[m]) / 2.0 for m in range(len(tmax))])

            anos+=1
        app = app / anos
        abt = abt / (anos * 12)
        per = abt / app * 58.93 if app > 0.0 else 0.0
        estaciones_df.loc[estaciones_df['Identificador'] == identificador, 'ABT'] = round(abt, 2)
        estaciones_df.loc[estaciones_df['Identificador'] == identificador, 'APP'] = round(app, 2)
        estaciones_df.loc[estaciones_df['Identificador'] == identificador, 'PER'] = round(per, 2)
        print(f"  ABT = {abt:.2f}, APP = {app:.2f}, PER = {per:.2f}")

    # Guardar el dataframe resultante en OUTPUT_FILE
    estaciones_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Datos de estaciones procesados guardados en {OUTPUT_FILE}")







        