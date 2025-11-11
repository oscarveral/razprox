from pathlib import Path

from bioclas import load_variables, load_fis

if __name__ == "__main__":
    RESOURCES_PATH = Path(__file__).parent.parent / "resources"
    VARIABLES_FILE = RESOURCES_PATH / "variables.json"
    FIS_ZONIFY_FILE = RESOURCES_PATH / "FIS_Zonify.json"
    FIS_BIOTEM_FILE = RESOURCES_PATH / "FIS_Biotem.json"

    # Cargar las variables difusas desde el archivo JSON
    variables = load_variables(VARIABLES_FILE)

    # Cargar el sistema de inferencia difusa para Zonify
    fis_zonify = load_fis(FIS_ZONIFY_FILE, variables)