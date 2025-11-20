""" Comparamos la salida FID3.5 con las tres zonas de vida inferidas por Zonify
"""

import pandas as pd
import time
import math
import numpy as np
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt

CONFIGS = Path(__file__).parent.parent / "configs"
INPUT_CSV = Path(__file__).parent.parent / "resources3" / "zonify_fused" / "zonify_fused_results.csv"
INPUT_FID = Path(__file__).parent.parent / "resources3" / "fid" / "sintetico" / "test.file"

def procesar_datos_texto(contenido_texto):
    """
    Procesa el contenido de texto crudo, extrae la tabla de datos
    y devuelve un DataFrame de Pandas con la zona de vida inferida.
    """
    data = []
    procesando_tabla = False
    
    # Dividimos el contenido en líneas
    lineas = contenido_texto.splitlines()
    
    for linea in lineas:
        linea_limpia = linea.strip()
        
        # 1. Detectar el inicio de la cabecera
        # Buscamos la línea que contiene "Example" y "ABT"
        if "Example" in linea_limpia and "ABT" in linea_limpia:
            procesando_tabla = True
            continue  # Saltamos la línea de cabecera para ir a los datos
            
        # 2. Procesar las líneas de datos
        if procesando_tabla:
            # Si la línea está vacía, terminamos
            if not linea_limpia:
                break
            
            # Las líneas de datos válidas deben empezar por 'x' (ej: x0, x1)
            if not linea_limpia.startswith('x'):
                break
            
            # Dividimos la línea por espacios en blanco
            partes = linea_limpia.split()
            
            # Estructura esperada de 'partes':
            # [0]: ID (x0)
            # [1]: ABT
            # [2]: APP
            # [3]: PER
            # [4]: || (Separador a ignorar)
            # [5]: Decision
            # [6]: Actual
            # [7]: * (Opcional, indicador de error, lo ignoramos)
            
            if len(partes) >= 6:
                fila = {
                    #'Example_ID': partes[0],
                    #'ABT': float(partes[1]),
                    #'APP': float(partes[2]),
                    #'PER': float(partes[3]),
                    # Saltamos partes[4] que es '||'
                    'Decision': partes[4],
                    # 'Actual': partes[6]
                }
                data.append(fila)

    # Crear el DataFrame
    df = pd.DataFrame(data)
    return df


df_fid = procesar_datos_texto(INPUT_FID.read_text())
df_zonify = pd.read_csv(INPUT_CSV)

df_merged = pd.merge(df_zonify, df_fid, left_index=True, right_index=True, how='inner')

def abreviar_zona(zona):
    if pd.isna(zona):
        return None
    return ''.join([palabra[0].upper() for palabra in zona.split('-')])

df_merged['Z1_abrev'] = df_merged['Z1'].apply(abreviar_zona)
df_merged['Z2_abrev'] = df_merged['Z2'].apply(abreviar_zona)
df_merged['Z3_abrev'] = df_merged['Z3'].apply(abreviar_zona)

# Comparamos la columna 'Decision' con las zonas de vida abreviadas
match_Z1 = df_merged['Decision'] == df_merged['Z1_abrev']
match_Z2 = df_merged['Decision'] == df_merged['Z2_abrev']
match_Z3 = df_merged['Decision'] == df_merged['Z3_abrev']

print("Resultados de la comparación entre FID3.5 y Zonify para fichero", INPUT_FID)
print(f"Total de muestras: {len(df_merged)}")
print(f"Coincidencias con Z1: {match_Z1.sum()} ({(match_Z1.sum() / len(df_merged)) * 100:.2f}%)")
print(f"Coincidencias con Z2: {match_Z2.sum()} ({(match_Z2.sum() / len(df_merged)) * 100:.2f}%)")
print(f"Coincidencias con Z3: {match_Z3.sum()} ({(match_Z3.sum() / len(df_merged)) * 100:.2f}%)")
print(f"Coincidencias totales (Z1, Z2 o Z3): {(match_Z1 | match_Z2 | match_Z3).sum()} ({((match_Z1 | match_Z2 | match_Z3).sum() / len(df_merged)) * 100:.2f}%)")