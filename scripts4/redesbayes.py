from pathlib import Path

from pgmpy.readwrite import BIFReader
from pgmpy.readwrite import BIFWriter
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

RESOURCES_PATH = Path(__file__).parent.parent / "resources4"

reader = BIFReader(RESOURCES_PATH / "consumo.bif")
model = reader.get_model()

var = reader.get_variables()
valores = reader.get_states()
edges = reader.get_edges()
cpds = reader.get_values()

print(f"Variables en el modelo:\n {var} \n")
print(f"Valores de la variables en el modelo:\n {valores} \n")
print(f"Aristas en el modelo:\n {edges} \n")
print(f"CPDs en el modelo:\n {cpds} \n")

model_infer = VariableElimination(model)

q = model_infer.query(variables=["Bono"], evidence={"Ventanas": "No", "Iluminacion": "No", "PlacasSolares": "No"})
print(f"Resultado de la consulta B | V=No, I=No, PS=No:\n {q} \n")

q = model_infer.query(variables=["Bono"], evidence={"Ventanas": "Si", "Iluminacion": "Si", "PlacasSolares": "No"})
print(f"Resultado de la consulta B | V=Si, I=Si, PS=No:\n {q} \n")
q = model_infer.query(variables=["Bono"], evidence={"Ventanas": "Si", "Iluminacion": "No", "PlacasSolares": "Si"})
print(f"Resultado de la consulta B | V=Si, I=No, PS=Si:\n {q} \n")
q = model_infer.query(variables=["Bono"], evidence={"Ventanas": "No", "Iluminacion": "Si", "PlacasSolares": "Si"})
print(f"Resultado de la consulta B | V=No, I=Si, PS=Si:\n {q} \n")