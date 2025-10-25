from bioclas.fuzzylogic.fuzzy_set import FuzzySet
from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable
from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter
import bioclas.fuzzylogic.mem_functions as mf
from bioclas.fuzzylogic.fuzzy_rule import FuzzyRule

if __name__ == "__main__":
    velocidad = FuzzyVariable("Velocidad", (0, 100), 0.1)
    baja = FuzzySet(
        "Baja",
        lambda x: mf.trimf(x, 10, 30, 50),
    )
    velocidad.add_fuzzyset(baja)

    frenado = FuzzyVariable("Frenado", (0, 50), 0.1)
    baja = FuzzySet("Baja", lambda x: mf.trimf(x, 10, 20, 30))
    frenado.add_fuzzyset(baja)

    regla1 = FuzzyRule()
    regla1.add_antecedent(velocidad, "Baja")
    regla1.set_consequent(frenado, "Baja")

    value, var, fs_name = regla1.eval([40])
    print(
        f"Degree of membership: {value} in variable '{var.name}' for fuzzy set '{fs_name}'"
    )
