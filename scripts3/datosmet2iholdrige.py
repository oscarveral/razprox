"""Script para transformar un dataframe de estaciones meteorológicas en indicadores de Holdridge."""

from pathlib import Path
import time
import numpy as np
import pandas as pd

# Configuración de rutas (Ajusta según tu estructura real)
BASE_DIR = Path(__file__).parent.parent # Asumiendo estructura scripts3/../
INPUT_FILE1 = BASE_DIR / "resources3" / "datosmet" / "datosmet_completos.csv" 
INPUT_FILE2 = BASE_DIR / "resources3" / "datosmet" / "datosmet_incompletos.csv"
OUTPUT_FOLDER = BASE_DIR / "resources3" / "indicadores"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def calcular_abt_estacion(df_input: pd.DataFrame) -> float:
    """Calcula la Biotemperatura Media Anual (ABT)."""
    # 1. Evitar SettingWithCopyWarning
    df = df_input.copy()
    
    # 2. Preprocesamiento
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['tmean'] = (df['tmax'] + df['tmin']) / 2
    
    # Clippear temperatura (Regla Holdridge: 0 < T < 30)
    df['tmean_clipped'] = df['tmean'].clip(lower=0, upper=30)

    # 3. Agrupar: Primero media mensual por año, luego media climática
    abt_monthly = df.groupby('Month')['tmean_clipped'].mean()
    
    # Asegurar índice 1-12
    abt_monthly = abt_monthly.reindex(range(1, 13))
    nans = abt_monthly.isna().sum()
    # Triplicar abt_monthly para facilitar interpolación circular
    abt_monthly = pd.concat([abt_monthly, abt_monthly, abt_monthly], ignore_index=True)

    # 4. Lógica de interpolación
    if nans > 3:
        return np.nan
    elif nans > 0:
        try:
            # Intento PCHIP (suave)
            abt_monthly = abt_monthly.interpolate(method='pchip')
        except Exception as e:
            print(f"PCHIP falló: {e}")
            
        # Tomar solo la parte central
        abt_monthly = abt_monthly[12:24]
    
    # 5. Validación final
    if abt_monthly.isna().sum() > 0:
        return np.nan
    
    # ABT es el PROMEDIO de los meses
    return np.clip(abt_monthly.mean(),0.375, 30)

def calcular_app_estacion(df_input: pd.DataFrame) -> float:
    """Calcula la Precipitación Total Anual (APP)."""
    df = df_input.copy()
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # 1. Calcular el PROMEDIO de esas sumas para obtener la "Normal Climática Mensual"
    app_monthly = df.groupby('Month')['prcp'].mean()
    
    app_monthly = app_monthly.reindex(range(1, 13))
    nans = app_monthly.isna().sum()
    app_monthly = pd.concat([app_monthly, app_monthly, app_monthly], ignore_index=True)

    # 2. Interpolación
    if nans > 3:
        return np.nan
    elif nans > 0:
        try:
            app_monthly = app_monthly.interpolate(method='pchip')
        except Exception as e:
            print(f"PCHIP falló: {e}")
            
        app_monthly = app_monthly[12:24]
    
    if app_monthly.isna().sum() > 0:
        return np.nan
    
    return np.clip(app_monthly.sum(), 62.5, 16000)

def procesar_dataset(ruta_archivo, nombre_salida):
    """Función auxiliar para procesar un archivo completo de forma optimizada."""
    if not ruta_archivo.exists():
        print(f"Archivo no encontrado: {ruta_archivo}")
        return

    print(f"Leyendo {ruta_archivo.name}...")
    tic = time.time()
    
    df = pd.read_csv(ruta_archivo, parse_dates=['Date'])
    
    # Lista para guardar resultados por estación
    resultados = []
    estaciones = df['StationID'].unique()
    
    print(f"Procesando {len(estaciones)} estaciones...")
    
    count_completos = 0
    count_desechados = 0
    
    for estacion in estaciones:
        # Filtramos una sola vez
        df_estacion = df[df['StationID'] == estacion]
        
        abt = calcular_abt_estacion(df_estacion)
        app = calcular_app_estacion(df_estacion)
        
        per = np.nan
        
        # Lógica de negocio Holdridge
        # Validamos APP
        if pd.notna(app):
            app = np.clip(app, a_min=62.5, a_max=16000)
            
        # Validamos PER
        if pd.notna(abt) and pd.notna(app) and app != 0:
            # Formula ratio evapotranspiración potencial
            per = (abt * 58.93) / app
            per = np.clip(per, a_min=0.125, a_max=32)
            count_completos += 1

        if pd.isna(abt) and pd.isna(app):
            count_desechados += 1
            continue
            
        resultados.append({
            'StationID': estacion,
            'ABT': abt,
            'APP': app,
            'PER': per
        })

    # Convertimos resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Añadir las latitudes y longitudes originales
    df_final = df[['StationID', 'Longitud', 'Latitud']].drop_duplicates().merge(
        df_resultados, on='StationID', how='right'
    )
    
    output_path = OUTPUT_FOLDER / nombre_salida
    df_final.to_csv(output_path, index=False)
    
    print(f"Guardado en {output_path}")
    print(f"Estaciones calculadas correctamente: {count_completos}/{len(estaciones)}")
    print(f"Estaciones desechadas por faltar todos los datos: {count_desechados}")
    print(f"Tiempo: {time.time() - tic:.2f} s\n")

# --- Ejecución Principal ---

if __name__ == "__main__":
    # Procesar archivo 1
    procesar_dataset(INPUT_FILE1, "indicadores_completos.csv")
    procesar_dataset(INPUT_FILE2, "indicadores_incompletos.csv")