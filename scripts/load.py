from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

RESOURCES_PATH = Path(__file__).resolve().parents[1] / "resources"
PENINSULA_PATH = RESOURCES_PATH / "ign" / "gadm41_ESP_2.shp"

try:
    peninsula = gpd.read_file(PENINSULA_PATH)

    fig, ax = plt.subplots(figsize=(80, 60))
    fig.patch.set_facecolor('white')  # Establecer el fondo de la figura en blanco

    peninsula.plot(ax=ax, color="gray", edgecolor="black", label="España")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Silueta de la Península Ibérica")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"No se pudo cargar el shapefile de la silueta: {e}")
