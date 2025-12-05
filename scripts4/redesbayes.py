from pathlib import Path

from pgmpy.readwrite import BIFReader
from pgmpy.readwrite import BIFWriter
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

RESOURCES_PATH = Path(__file__).parent.parent / "resources4"

reader = BIFReader(RESOURCES_PATH / "consumo.bif")