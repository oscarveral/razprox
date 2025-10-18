import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet

def min_t_norm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
        def min_membership(x):
            return np.minimum(a.mf(x), b.mf(x))

        return FuzzySet(
            name=f"MinTNorm({a.name}, {b.name})",
            membership_function=min_membership,
        )

def prod_t_norm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
        def prod_membership(x):
            return a.mf(x) * b.mf(x)

        return FuzzySet(
            name=f"ProdTNorm({a.name}, {b.name})",
            membership_function=prod_membership,
        )