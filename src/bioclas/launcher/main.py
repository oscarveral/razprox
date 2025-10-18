from bioclas.fuzzylogic.fuzzy_set import FuzzySet

from bioclas.fuzzylogic.mem_functions import trapmf, trimf, sigmf
from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter

if __name__ == "__main__":

    # Probar todas las funciones de membresía y graficarlas
    fplotter = FuzzyPlotter()
    fplotter.domain = (0, 20)

    trapezoidal_set = FuzzySet(
        "Trapezoidal Set", lambda x: trapmf(x, 0, 0, 2, 5)
    )
    fplotter.add_fuzzy_set(trapezoidal_set)

    triangular_set = FuzzySet(
        "Triangular Set", lambda x: trimf(x, 2, 5, 8)
    )
    fplotter.add_fuzzy_set(triangular_set)

    sigmoidal_set = FuzzySet(
        "Sigmoidal Set", lambda x: sigmf(x, 3, 10)
    )
    fplotter.add_fuzzy_set(sigmoidal_set)

    fplotter.plot(num_points=200, title="Membership Functions Example")


    # Probar negacion, t-norma y t-conorma
    from bioclas.fuzzylogic.complement import complement_minus
    from bioclas.fuzzylogic.t_norm import min_t_norm, prod_t_norm
    from bioclas.fuzzylogic.t_conorm import max_t_conorm, sum_t_conorm

    s = FuzzySet(
        "Set S", lambda x: trimf(x, 0, 5, 10)
    )
    not_s = complement_minus(s)
    
    plotter = FuzzyPlotter()
    plotter.domain = (0, 10)
    plotter.add_fuzzy_set(s)
    plotter.add_fuzzy_set(not_s)
    plotter.plot(title="Complement Example")

    and_set = prod_t_norm(s, not_s)
    or_set = sum_t_conorm(s, not_s)
    plotter2 = FuzzyPlotter()
    plotter2.domain = (0, 10)
    plotter2.add_fuzzy_set(and_set)
    plotter2.add_fuzzy_set(or_set)
    plotter2.add_fuzzy_set(s)
    plotter2.add_fuzzy_set(not_s)
    plotter2.plot(title="T-Conorm Example")


