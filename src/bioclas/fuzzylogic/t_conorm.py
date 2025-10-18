import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet

def max_t_conorm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
        def max_membership(x):
            return np.maximum(a.mf(x), b.mf(x))

        return FuzzySet(
            name=f"MaxTConorm({a.name}, {b.name})",
            membership_function=max_membership,
        )

def sum_t_conorm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
        def sum_membership(x):
            return a.mf(x) + b.mf(x) - a.mf(x) * b.mf(x)

        return FuzzySet(
            name=f"SumTConorm({a.name}, {b.name})",
            membership_function=sum_membership,
        )