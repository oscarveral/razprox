"""Script para transfrormar los resultados de Zonify a un formato compatible con FID3.5.
"""
import math
from pathlib import Path
import json
import pandas as pd
import argparse

CONFIGS = Path(__file__).parent.parent / "configs"
INPUT = Path(__file__).parent.parent / "resources3" / "indicadores" / "all.csv"
OUTPUT = Path(__file__).parent.parent / "resources3" / "fid" / "all"
Path.mkdir(OUTPUT, parents=True, exist_ok=True)

VARS = ['ABT', 'APP']
N_VARS = len(VARS)

variables_json = CONFIGS / "variables.json"
variables_dict = json.load(open(variables_json, "r"))

# Mapear las zonas de vida a sus iniciales
zonas_abreviaturas = {}
for zona in variables_dict['ZonaDeVida']['Etiquetas'].keys():
    # Caracteres en mayuscula de cada palabra. Matorral-Muy-Humedo -> MMH
    abreviatura = ''.join([palabra[0].upper() for palabra in zona.split('-')])
    zonas_abreviaturas[zona] = abreviatura

def dat(input, OUTPUT_FOLDER: Path):
    df = pd.read_csv(input)
    n_samples = {}

    def transform(var, x):
        dominio = variables_dict[var]['Dominio']
        constante = variables_dict[var]['Escala']['Constante']
        x = math.log2(x/constante)
        return (x - dominio[0]) / (dominio[1] - dominio[0])

    count = 0
    with open(OUTPUT_FOLDER / "all.dat", "w") as f:
        for _, row in df.iterrows():
            for var in VARS:
                dominio = variables_dict[var]['Dominio']
                value = row[var]
                # If valie is na, put -1
                if pd.isna(value):
                    value = -1
                else:
                    value = transform(var, value)
                f.write(f"{value:.3f} ")
            f.write("DP 1\n")
            
            count += 1
        # Append count at the beginning of the file
    with open(OUTPUT_FOLDER / "all.dat", "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"{count} {N_VARS}\n" + content)
            
dat(INPUT, OUTPUT)

        


