"""Script para transfrormar los resultados de Zonify a un formato compatible con FID3.5.
"""
import math
from pathlib import Path
import json
import pandas as pd
import argparse

CONFIGS = Path(__file__).parent.parent / "configs"
INPUT = Path(__file__).parent.parent / "resources3" / "zonify_fused" / "zonify_fused_results.csv"
OUTPUT = Path(__file__).parent.parent / "resources3" / "fid"
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

def attrs(output_folder: Path): 
    with open(output_folder / "data.attrs", "w") as f:
        f.write(f"{N_VARS}\n")
        for i, var in enumerate(VARS):
            dominio = variables_dict[var]['Dominio']
            transform = lambda x: (x - dominio[0]) / (dominio[1] - dominio[0])
            etiquetas = variables_dict[var]['Etiquetas']
            base = variables_dict[var]['Escala']['Base']
            constante = variables_dict[var]['Escala']['Constante']

            f.write(f"{var} 1 {len(etiquetas)} 0 1\n")

            # Normalizar los intervalos de las etiquetas al rango [0, 1]
            midpoints = []
            ekeys = list(etiquetas.keys())
            for etiqueta in ekeys:
                intervalo = etiquetas[etiqueta]
                intervalo_norm = [transform(intervalo[0]), transform(intervalo[1])]
                midpoint = (intervalo_norm[0] + intervalo_norm[1]) / 2
                midpoints.append(midpoint)

            for i, (etiqueta, intervalo) in enumerate(etiquetas.items()):
                if i == 0:
                    f.write(f"{etiqueta} 0 0 {midpoints[0]:.2f} {midpoints[1]:.2f}\n")
                elif i == len(etiquetas) - 1:
                    f.write(f"{etiqueta} {midpoints[-2]:.2f} {midpoints[-1]:.2f} 1 1\n")
                else:
                    f.write(f"{etiqueta} {midpoints[i-1]:.2f} {midpoints[i]:.2f} {midpoints[i]:.2f} {midpoints[i+1]:.2f}\n")
        
        zonas = variables_dict['ZonaDeVida']
        etiquetas = zonas['Etiquetas']
        n_zonas = len(etiquetas)
        
        f.write(f"\n0 {n_zonas}\n")

        for abreviatura in zonas_abreviaturas.values():
            f.write(f"{abreviatura}\n")

def dat(INPUT, OUTPUT_FOLDER: Path):
    df = pd.read_csv(INPUT)
    n_samples = {}

    def transform(var, x):
        dominio = variables_dict[var]['Dominio']
        constante = variables_dict[var]['Escala']['Constante']
        x = math.log2(x/constante)
        return (x - dominio[0]) / (dominio[1] - dominio[0])
    
    for _, row in df.iterrows():
        zona = row['Z1']
        if pd.isna(zona):
            continue
        if zona not in n_samples:
            n_samples[zona] = 0
        n_samples[zona] += 1 

    count = 0
    with open(OUTPUT_FOLDER / "data.dat", "w") as f:
        for _, row in df.iterrows():
            if pd.isna(row['Z1']):
                continue
            for var in VARS:
                dominio = variables_dict[var]['Dominio']
                f.write(f"{transform(var, row[var]):.3f} ")
            f.write(f"{zonas_abreviaturas[row['Z1']]} ")
            weight = 1 / (math.sqrt(n_samples[row['Z1']]))
            f.write(f"{weight:.6f}\n")
            
            count += 1
        # Append count at the beginning of the file
    with open(OUTPUT_FOLDER / "data.dat", "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"{count} {N_VARS}\n" + content)

    print("Número de muestras por zona de vida:")
    for zona, count in n_samples.items():
        print(f"  {zona}: {count}")

            
attrs(OUTPUT)
dat(INPUT, OUTPUT)

        


