from bioclas.fuzzylogic.fuzzy_set import FuzzySet

def complement_minus(a: FuzzySet) -> FuzzySet:
    def neg_membership(x):
        return 1 - a.mf(x)

    return FuzzySet(
        name=f"ComplementMinus({a.name})",
        membership_function=neg_membership,
    )