"""Script para transformar un dataframe de estaciones meteorológicas en indicadores de Holdridge."""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calcular_indicadores_holdridge_estacion(df: pd.DataFrame) -> dict:
    """Recibe un dataframe con datos meteorológicos y calcula los indicadores de Holdridge."""
    """Columnas esperadas: ['Date', 'StationID', 'Latitud', 'Longitud', 'tmax', 'tmin', 'prcp']"""
    """StationID, Latitud, Longitud son constantes por estación."""
    # Extraer los años únicos en el dataframe
    años = df['Date'].dt.year.unique()

    # Calcular ABT
    """
    El procedimiento es el siguiente:
    Calcular la temperatura media de cada par (año,mes) y clippearla entre 0 y 30.
    Si ese dato no se puede calcular (NA), dejarlo como NA.
    Luego, calcular la ABT de cada mes como la media de todos los datos disponibles para ese mes.
    Si un mes no tiene datos para ningun año, dejarlo como NA.
    Si el año tiene mas de 3 meses con NA, dejar la ABT anual como NA.
    Si no, interpolar con polinomio de grado 4 los meses faltantes.
    """
    # TODO
    # bt_mensual = []
    # for año in años:
    #     df_año = df[df['Date'].dt.year == año]
    #     for mes in range(1, 13):
    #         df_mes = df_año[df_año['Date'].dt.month == mes]
    #         if df_mes.empty:
    #             bt_mensual.append({'Año': año, 'Mes': mes, 'BT_mensual': np.nan})
    #             continue
    #         t_media = (df_mes['tmax'] + df_mes['tmin']) / 2
    #         t_media_clipped = np.clip(t_media, 0, 30)
    #         bt_mensual.append({'Año': año, 'Mes': mes, 'BT_mensual': t_media_clipped.mean()})