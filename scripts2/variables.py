"""Script para cargar las variables difusas y guardar gráficas de sus funciones de pertenencia."""

from bioclas import load_variables
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    CONFIGS = Path(__file__).parent.parent / "configs"
    VARIABLES_FILE = CONFIGS / "variables.json"
    OUTPUT_FOLDER = Path(__file__).parent.parent / "output" / "variable_membership_plots"
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Cargar las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)

    for var_name, variable in variables.items():
        variable.plotter().save_plot(
            filepath = OUTPUT_FOLDER / f"{var_name}_membership_functions.png",
            title = f"Membership Functions for Variable: {var_name}",
            xlabel = "Universe of Discourse",
            ylabel = "Membership Degree",
            step = 0.01
        )
