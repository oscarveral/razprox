from bioclas.fuzzylogic.fuzzy_set import FuzzySet
from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter
from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable
import bioclas.fuzzylogic.mem_functions as mf

if __name__ == "__main__":
    temperatura = FuzzyVariable("Temperatura", (-10, 40), 0.1)
    baja = FuzzySet(
        "Baja",
        lambda x: mf.trapmf(x, -10, -10, 0, 10),
    )
    media = FuzzySet(
        "Media",
        lambda x: mf.trapmf(x, 0, 10, 15, 25),
    )
    alta = FuzzySet(
        "Alta",
        lambda x: mf.trapmf(x, 15, 25, 40, 40),
    )

    temperatura.add_fuzzyset(baja)
    temperatura.add_fuzzyset(media)
    temperatura.add_fuzzyset(alta)
    temperatura.plot()
