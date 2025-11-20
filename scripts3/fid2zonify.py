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
INPUT_FID = Path(__file__).parent.parent / "resources3" / "fid" / "normal" / "test.file"
OUTPUT = Path(__file__).parent.parent / "resources3" / "fidcolor" / "normal"
Path.mkdir(OUTPUT, parents=True, exist_ok=True)

def procesar_datos_texto(contenido_texto):
    """
    Procesa el contenido de texto crudo, extrae la tabla de datos
    y devuelve un DataFrame de Pandas.
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
                    'Decision': partes[5],
                    'Actual': partes[6]
                }
                data.append(fila)

    # Crear el DataFrame
    df = pd.DataFrame(data)
    return df

def plot_confusion_matrix_with_diagonal_highlight(
    confusion_matrix: pd.DataFrame, 
    filename: str,
    title: str = "Matriz de Confusión", 
    cmap: str = "Blues", # Puedes elegir otro mapa de color, por ejemplo "YlGnBu", "viridis"
    diag_color: str = "lightcoral", # Color para la diagonal
    annot: bool = True, # Mostrar valores en las celdas
    fmt: str = "d", # Formato de anotación, "d" para enteros, ".1f" para flotantes
    figsize: tuple = (8, 6) # Tamaño de la figura
):
    """
    Genera y guarda una imagen PNG de una matriz de confusión de pandas.crosstab,
    resaltando la diagonal con un color diferente.

    Args:
        confusion_matrix (pd.DataFrame): La matriz de confusión generada por pd.crosstab.
                                        Debe tener los mismos índices y columnas para clases reales y predichas.
        title (str): Título de la imagen.
        filename (str): Nombre del archivo PNG a guardar.
        cmap (str): Mapa de color de Seaborn para las celdas no diagonales.
        diag_color (str): Color para resaltar las celdas de la diagonal.
        annot (bool): Si es True, mostrar los valores numéricos en cada celda.
        fmt (str): Formato para las anotaciones (por ejemplo, 'd' para enteros, '.1f' para flotantes).
        figsize (tuple): Tamaño de la figura (ancho, alto) en pulgadas.
    """
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(confusion_matrix, annot=False, fmt=fmt, cmap=cmap, cbar=True, linewidths=.5, linecolor='black')

    # Recorrer las celdas de la diagonal y aplicar el color especial
    for i in range(len(confusion_matrix)):
        # Obtener las coordenadas de la celda de la diagonal
        # El orden de las coordenadas para matplotlib/seaborn es (columna, fila)
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor=diag_color, lw=3))
        
        # Opcional: Para cambiar el color de fondo de la celda en lugar del borde
        # Usamos ax.text para volver a colocar las anotaciones si annot=False inicialmente
        # o para asegurar que el color de fondo se aplica correctamente.
        # Si prefieres que el fondo sea el diag_color:
        ax.add_patch(plt.Rectangle((i, i), 1, 1, facecolor=diag_color, alpha=0.6))
    
    # Si annot es True, volvemos a añadir las anotaciones después de pintar la diagonal
    # para que se superpongan correctamente
    if annot:
        for text in ax.texts:
            text.set_color('black') # Asegurarse de que el texto sea legible
            
        # Re-aplicar anotaciones
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                text_val = confusion_matrix.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f'{text_val:{fmt}}', 
                        ha='center', va='center', color='black', fontsize=10) # Ajusta fontsize si es necesario

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Clase Predicha", fontsize=12)
    ax.set_ylabel("Clase Verdadera", fontsize=12)
    plt.tight_layout() # Ajusta el diseño para evitar recortes
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Imagen '{filename}' generada y guardada correctamente.")
    plt.close() # Cierra la figura para liberar memoria

df_fid = procesar_datos_texto(INPUT_FID.read_text())
df_zonify = pd.read_csv(INPUT_CSV)

# Check matching lengths
if len(df_fid) != len(df_zonify):
    raise ValueError(f"Los archivos de entrada no tienen la misma cantidad de muestras: {len(df_fid)} vs {len(df_zonify)}")

# Combinar los DataFrames basados en el índice
df_combined = pd.concat([df_fid.reset_index(drop=True), df_zonify.reset_index(drop=True)], axis=1)
df_combined.to_csv(OUTPUT / "fid2zonify_results.csv", index=False)

# Matriz de confusión entre Decision y Actual
# Usar en los ejes el mismo orden de etiquetas
etiquetas = sorted(df_combined['Actual'].dropna().unique())
confusion_matrix = pd.crosstab(df_combined['Actual'], df_combined['Decision'], rownames=['Actual'], colnames=['Decision'], dropna=False).reindex(index=etiquetas, columns=etiquetas, fill_value=0)
confusion_matrix.to_csv(OUTPUT / "confusion_matrix.csv")

plot_confusion_matrix_with_diagonal_highlight(
    confusion_matrix, 
    filename=str(OUTPUT / "confusion_matrix.png"),
    title="Matriz de Confusión entre Decision y Actual",
    cmap="Blues",
    diag_color="lightcoral",
    annot=True,
    fmt="d",
    figsize=(10, 10)
)
# # Save as png with heatmap
# import seaborn as sns
# import matplotlib.pyplot as plt 
# plt.figure(figsize=(20, 20))
# # Diagonal in a different color. Paint over the diagonal
# mask = np.eye(confusion_matrix.shape[0], dtype=bool)


# sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=.5, linecolor='black')
# plt.title('Matriz de Confusión entre Decision y Actual')
# plt.savefig(OUTPUT / "confusion_matrix.png") 


