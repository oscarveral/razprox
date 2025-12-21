import pandas as pd
import numpy as np
from pathlib import Path

# Intentamos importar pgmpy y verificar su versión
try:
    import pgmpy
    print(f"Versión de pgmpy detectada: {pgmpy.__version__}")
except ImportError:
    print("Error crítico: pgmpy no está instalado.")
    exit()

# ---------------------------------------------------------
# BLOQUE DE IMPORTACIÓN A PRUEBA DE FALLOS
# ---------------------------------------------------------
from pgmpy.models import BayesianNetwork

# Importación de Estructura
try:
    from pgmpy.estimators import HillClimbSearch
except ImportError:
    # Intento alternativo para versiones muy antiguas
    from pgmpy.estimators import HillClimbSearch

# Importación de Parámetros
try:
    from pgmpy.estimators import MaximumLikelihoodEstimator
except ImportError:
    from pgmpy.estimators import MaximumLikelihoodEstimator

# Importación del Score (BicScore)
# Intentamos buscarlo en sub-módulos específicos para saltar el error del __init__
ScoreClass = None
try:
    # Intento 1: Estándar
    from pgmpy.estimators import BicScore
    ScoreClass = BicScore
    print("Importado BicScore desde pgmpy.estimators")
except ImportError:
    try:
        # Intento 2: Ruta directa al archivo (común en v0.1.23+)
        from pgmpy.estimators.BIC import BicScore
        ScoreClass = BicScore
        print("Importado BicScore desde pgmpy.estimators.BIC")
    except ImportError:
        print("Aviso: No se pudo importar BicScore. Asegúrate de tener una versión compatible de pgmpy.")

# ---------------------------------------------------------
# Paso previo: Cargar los datos
# ---------------------------------------------------------
resources_path = Path(__file__).parent.parent / 'resources4'
filename = resources_path / 'sample_energy.csv'

try:
    data = pd.read_csv(filename)
    print(f"Datos cargados correctamente. Filas: {len(data)}, Columnas: {list(data.columns)}")
except FileNotFoundError:
    print(f"Error: No se encuentra el archivo '{filename}'. Asegúrate de que esté en la carpeta.")
    # Generamos datos dummy solo para que el ejemplo sea funcional si copias y pegas sin el archivo
    import numpy as np
    np.random.seed(42)
    data = pd.DataFrame({
        'EUI': np.random.randint(1, 6, 100),
        'area': np.random.randint(1, 4, 100),
        'students': np.random.randint(1, 4, 100),
        'floors': np.random.randint(1, 3, 100),
        'event': np.random.randint(0, 2, 100)
    })
    print("Usando datos simulados para el ejemplo...")

# ---------------------------------------------------------
# a) Aprendizaje de la Red Bayesiana (Estructura y Parámetros)
# ---------------------------------------------------------

print("\n--- a) Aprendizaje de la Estructura ---")
# Utilizamos HillClimbSearch, un algoritmo de búsqueda heurística común.
# BicScore (Criterio de Información Bayesiano) se usa para evaluar qué tan bien se ajusta la estructura a los datos,
# penalizando la complejidad para evitar sobreajuste.
hc = HillClimbSearch(data)
best_model_structure = hc.estimate(scoring_method=BicScore(data))

print("Aristas (enlaces) aprendidos:")
print(best_model_structure.edges())

# Creamos el modelo de Red Bayesiana con la estructura aprendida
model = BayesianNetwork(best_model_structure.edges())

print("\n--- a) Aprendizaje de los Parámetros ---")
# Aprendemos las Tablas de Probabilidad Condicional (CPDs) usando Máxima Verosimilitud (MLE)
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Imprimimos las CPDs para verificar (opcional)
for cpd in model.get_cpds():
    print(f"CPD para la variable '{cpd.variable}':")
    # Imprimimos solo las dimensiones para no saturar la pantalla
    print(f"  Forma: {cpd.values.shape}") 

# ---------------------------------------------------------
# b) Realizar la predicción de la variable EUI
# ---------------------------------------------------------

print("\n--- b) Predicción de la variable EUI ---")

# Para predecir 'EUI', necesitamos el resto de datos como evidencia.
# Eliminamos la columna 'EUI' del conjunto de datos de prueba.
X_test = data.drop(columns=['EUI'])

# La función predict de pgmpy calcula el estado más probable de la variable objetivo
# dadas las variables de evidencia para cada fila.
y_pred = model.predict(X_test)

print("\nEjemplo de predicciones generadas (primeras 5 filas):")
print(y_pred.head())

# (Opcional) Evaluar la precisión básica comparando con los datos reales
# Nota: Esto es una evaluación sobre los mismos datos de entrenamiento (resustitución),
# idealmente se usaría un conjunto de test separado.
accuracy = (y_pred['EUI'] == data['EUI']).mean()
print(f"\nExactitud (Accuracy) sobre los datos proporcionados: {accuracy:.2%}")