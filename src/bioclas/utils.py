import csv
from functools import partial
import json
from pathlib import Path

from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter

from .fuzzylogic import FuzzyVariable, FuzzyVariableQualitative, FuzzySet, FIS
from .fuzzylogic.mem_functions import trapmf, trimf

def load_variables(file_path: Path) -> dict:
    """Carga variables difusas desde un fichero .json.

    Cada variable se describe con un nombre y un diccionario de atributos. Entre los atributos tenemos:
    - Tipo: Cualitativa o Cuantitativa.
    - Dominio: Rango de valores posibles [a, b].
    - Escala: Tipo de escala (Lineal o Exponencial) y parámetros asociados.
              En caso de escala exponencial, se incluye la base 'B' y una constante multiplicativa 'K'. 
              El dominio real se calcula como [B^a, B^b] * K.
    - Etiquetas: Diccionario que mapea etiquetas a subrangos dentro del dominio.
    Se espera que las etiquetas cubran todo el dominio sin solapamientos de forma contigua.
    La función de pertenencia aplicada es triangular entre los puntos medios de las etiquetas adyacentes,
    excepto en los extremos del dominio donde es trapezoidal.

    Ejemplo:

    "ABT": {
            "Tipo": "Cuantitativa",
            "Dominio": [-1.0, 5.33],
            "Escala": {
                "Tipo": "Exponencial",
                "Base": 2.0,
                "Constante": 0.75
            },
            "Etiquetas": {
                "0a1.5": [-1.0, 1.0],
                "1.5a3": [1.0, 2.0],
                "3a6": [2.0, 3.0],
                "6a12": [3.0, 4.0],
                "12a18": [4.0, 4.5],
                "18a24": [4.5, 5.0],
                "24a30": [5.0, 5.33]
            }
        }

    Args:
        file_path (Path): Ruta al archivo de texto.

    Returns:
        dict: Diccionario con las variables cargadas y definidas.
    """
    # Leer el archivo JSON y cargar las variables
    with file_path.open('r') as file:
        variables = json.load(file)

        fuzzy_vars = {}

        for var_name, attributes in variables.items():
            if not isinstance(attributes, dict):
                raise ValueError(f"Atributos de la variable '{var_name}' deben ser un diccionario.")
            if ("Tipo" not in attributes):
                raise ValueError(f"La variable '{var_name}' no tiene definido el atributo 'Tipo'.")
            if ("Etiquetas" not in attributes):
                raise ValueError(f"La variable '{var_name}' no tiene definido el atributo 'Etiquetas'.")
            
            var_type = attributes["Tipo"]

            if var_type not in ["Cualitativa", "Cuantitativa"]:
                raise ValueError(f"Tipo de variable '{var_name}' inválido: {var_type}")
            if var_type == "Cuantitativa":
                fuzzy_var = __process_quantitative_variable(var_name, attributes)
            else:  # Cualitativa
                fuzzy_var = __process_qualitative_variable(var_name, attributes)

            fuzzy_vars[var_name] = fuzzy_var

        return fuzzy_vars

def __process_quantitative_variable(var_name: str, attributes: dict) -> FuzzyVariable:
    """Procesa una variable cuantitativa y crea una instancia de FuzzyVariable.

    Args:
        var_name (str): Nombre de la variable.
        attributes (dict): Diccionario con los atributos de la variable.

    Returns:
        FuzzyVariable: Instancia de la variable difusa creada.
    """
    if ("Dominio" not in attributes):
        raise ValueError(f"La variable '{var_name}' no tiene definido el atributo 'Dominio'.")
    if ("Escala" not in attributes):
        raise ValueError(f"La variable '{var_name}' no tiene definido el atributo 'Escala'.")
    
    var_domain = attributes["Dominio"]
    var_scale = attributes["Escala"]
    var_labels = attributes["Etiquetas"]

    if not hasattr(var_domain, "__len__") or len(var_domain) != 2 or var_domain[0] >= var_domain[1]:
        raise ValueError(f"Dominio de variable '{var_name}' inválido: {var_domain}")
    if not isinstance(var_scale, dict):
        raise ValueError(f"Escala de variable '{var_name}' debe ser un diccionario.")
    if "Tipo" not in var_scale:
        raise ValueError(f"Escala de variable '{var_name}' no tiene definido el atributo 'Tipo'.")
    if var_scale["Tipo"] not in ["Lineal", "Exponencial"]:
        raise ValueError(f"Tipo de escala de variable '{var_name}' inválido: {var_scale['Tipo']}")
    if var_scale["Tipo"] == "Exponencial":
        if "Base" not in var_scale:
            raise ValueError(f"Escala exponencial de variable '{var_name}' no tiene definido el atributo 'Base'.")
        if "Constante" not in var_scale:
            raise ValueError(f"Escala exponencial de variable '{var_name}' no tiene definido el atributo 'Constante'.")
    if not isinstance(var_labels, dict) or not var_labels:
        raise ValueError(f"Etiquetas de variable '{var_name}' deben ser un diccionario no vacío.")
    
    fuzzy_var = FuzzyVariable(name=var_name, interval=tuple(var_domain))

    # Recorrer las etiquetas creando conjuntos difusos triangulares con maximos en los puntos medios.
    # Los puntos medios se calculan entre los rangos definidos en las etiquetas.
    # El primero y el último serán trapezoidales.

    mid_points = []
    for label, range_vals in var_labels.items():
        if not isinstance(range_vals, list) or len(range_vals) != 2:
            raise ValueError(f"Rango de etiqueta '{label}' en variable '{var_name}' inválido: {range_vals}")
        mid = (range_vals[0] + range_vals[1]) / 2.0
        mid_points.append(mid)

    for i, (label, range_vals) in enumerate(var_labels.items()):
        if i == 0:
            a = range_vals[0]
            b = range_vals[0]
            c = mid_points[i]
            d = mid_points[i + 1]
            fuzzy_var.add_fuzzyset(
                FuzzySet.trapezoidal(
                    name=label,
                    a=a, b=b, c=c, d=d
                )
            )                
        elif i == len(var_labels) - 1:
            a = mid_points[i - 1]
            b = mid_points[i]
            c = range_vals[1]
            d = range_vals[1]
            fuzzy_var.add_fuzzyset(
                FuzzySet.trapezoidal(
                    name=label,
                    a=a, b=b, c=c, d=d
                )
            )
        else:
            a = mid_points[i - 1]
            b = mid_points[i]
            c = mid_points[i + 1]
            fuzzy_var.add_fuzzyset(
                FuzzySet.triangular(name=label, a=a, b=b, c=c)
            )

    print(f"Variable difusa cuantitativa procesada: {var_name} con dominio {var_domain}")
    return fuzzy_var

def __process_qualitative_variable(var_name: str, attributes: dict) -> FuzzyVariable:
    """Procesa una variable cualitativa y crea una instancia de FuzzyVariable.

    Args:
        var_name (str): Nombre de la variable.
        attributes (dict): Diccionario con los atributos de la variable.

    Returns:
        FuzzyVariable: Instancia de la variable difusa creada.
    """
    var_labels = attributes["Etiquetas"]

    if not isinstance(var_labels, dict) or not var_labels:
        raise ValueError(f"Etiquetas de variable '{var_name}' deben ser un diccionario no vacío.")
    
    fuzzy_var = FuzzyVariableQualitative(name=var_name, interval=(0, len(var_labels) - 1))

    for i, (label, color) in enumerate(var_labels.items()):
        a = i - 1 if i > 0 else i
        b = i
        c = i + 1 if i < len(var_labels) - 1 else i
        fuzzy_var.add_color_fuzzyset(
            FuzzySet.triangular(
                name=label,
                a=a, b=b, c=c
            ),
            color=color
        )

    print(f"Variable difusa cualitativa procesada: {var_name} con dominio {[0, len(var_labels) - 1]}")
    return fuzzy_var

def load_fis(file_path: Path, variables: dict[str, FuzzyVariable]) -> FIS:
    """Carga un sistema de inferencia difusa (FIS) desde un fichero .json.

    Args:
        file_path (Path): Ruta al archivo de texto.

    Returns:
        FIS: Instancia del sistema de inferencia difusa cargado.
    """
    with file_path.open('r') as file:
        fis_data = json.load(file)


        a_vars_names = fis_data.get("a_variables", [])
        c_var_name = fis_data.get("c_variable", None)
        rules = fis_data.get("rules", {})

        if not a_vars_names:
            raise ValueError("No se han definido variables antecedentes en el FIS.")
        if c_var_name is None:
            raise ValueError("No se ha definido la variable consecuente en el FIS.")
        if c_var_name not in variables:
            raise ValueError(f"Variable consecuente '{c_var_name}' no encontrada entre las variables cargadas.")
        if not rules:
            raise ValueError("No se han definido reglas en el FIS.")

        fis = FIS(
            antecedents=[variables[var_name] for var_name in a_vars_names if var_name in variables],
            consequent=variables[c_var_name]
        )

        for rule_n, rule in rules.items():
            antecedents = rule["antecedentes"]
            consequent_fs_name = rule["consecuente"][c_var_name]
            fis.add_rule(rule_n, antecedents, consequent_fs_name)
            print(f"\tRegla '{rule_n}' añadida al FIS.")
        return fis
    
def load_geogrid(file_path: Path) -> list[tuple]:
    """Carga los datos de la cuadricula geográfica desde un fichero .csv.

    Args:
        file_path (Path): Ruta al archivo de texto.

    Returns:
        list[tuple]: Lista de tuplas con los datos de la cuadricula: Longitud, Latitud, Altitud, APP.
    """
    geogrid_data = []
    with file_path.open('r') as file:
        reader = csv.DictReader(file, delimiter=';')

        for row in reader:
            # Procesar cada fila y añadirla a la lista
            # Suponiendo que la primera columna es un identificador único
            geogrid_data.append((
                float(row['X'].replace(',', '.')),
                float(row['Y'].replace(',', '.')),
                float(row['ELEVA'].replace(',', '.')),
                float(row['PRECIPITA'].replace(',', '.'))
            ))
    return geogrid_data