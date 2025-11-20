"""Script para inferir el color de las muestras de FID3.5
"""

import math
from pathlib import Path
import json
import pandas as pd 

CONFIGS = Path(__file__).parent.parent / "configs"
INPUT = Path(__file__).parent.parent / "resources3" / "fidcolor" / "normal" / "fid2zonify_results.csv"
OUTPUT = Path(__file__).parent.parent / "resources3" / "fidcolor" / "normal"
Path.mkdir(OUTPUT, parents=True, exist_ok=True)

def load_lifezones(configs_folder: Path):
    variables_json = configs_folder / "variables.json"
    variables_dict = json.load(open(variables_json, "r"))

    # Mapear las zonas de vida a sus iniciales
    zonas2abreviaturas = {}
    abreviaturas2rgb = {}
    for zona, color in variables_dict['ZonaDeVida']['Etiquetas'].items():
        # Caracteres en mayuscula de cada palabra. Matorral-Muy-Humedo -> MMH
        abreviatura = ''.join([palabra[0].upper() for palabra in zona.split('-')])
        zonas2abreviaturas[abreviatura] = zona
        abreviaturas2rgb[abreviatura] = color

    return zonas2abreviaturas, abreviaturas2rgb

z2a, a2rgb = load_lifezones(CONFIGS)

df = pd.read_csv(INPUT)
df.drop(columns=['Z1', 'Z2', 'Z3'], inplace=True, errors='ignore')

# Recorrer el dataframe sustituyendo las columnas RGB por las dadas segun la columna Decision
rgb_list = []
for _, row in df.iterrows():
    decision = row['Decision']
    if pd.isna(decision) or decision not in a2rgb:
        rgb_list.append((0, 0, 0))  # Color negro para decisiones no válidas
    else:
        rgb_list.append(a2rgb[decision])

df['r'] = [rgb[0] for rgb in rgb_list]
df['g'] = [rgb[1] for rgb in rgb_list]
df['b'] = [rgb[2] for rgb in rgb_list]

df.to_csv(OUTPUT / "fid2zonify_colored.csv", index=False)