from functools import partial

import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet


def get_fuzzy_family(name: str, **kwargs) -> dict:
    """Retrieve a fuzzy logic family by name.

    Args:
        name (str): The name of the fuzzy logic family.
        **kwargs: Additional parameters for specific families.

    Returns:
        dict: A dictionary containing the fuzzy logic operations for the specified family.
    """

    FAMILIES = {
        "minmax": {
            "t_norm": min_t_norm,
            "t_conorm": max_t_conorm,
            "complement": complement_minus,
            "params": None,
        },
        "algebraic": {
            "t_norm": prod_t_norm,
            "t_conorm": sum_t_conorm,
            "complement": complement_minus,
            "params": None,
        },
        "drastic": {
            "t_norm": drastic_t_norm,
            "t_conorm": drastic_t_conorm,
            "complement": complement_minus,
            "params": None,
        },
        "dubois-prade": {
            "t_norm": dubois_prade_t_norm,
            "t_conorm": dubois_prade_t_conorm,
            "complement": complement_minus,
            "params": ["p"],
        },
        "yager": {
            "t_norm": yager_t_norm,
            "t_conorm": yager_t_conorm,
            "complement": complement_yager,
            "params": ["p"],
        },
        "schweizer-sklar": {
            "t_norm": schweizer_sklar_t_norm,
            "t_conorm": schweizer_sklar_t_conorm,
            "complement": complement_minus,
            "params": ["p"],
        },
    }

    if name not in FAMILIES:
        raise ValueError(
            f"Fuzzy family '{name}' is not recognized. Options are: {list(FAMILIES.keys())}"
        )

    fam = FAMILIES[name]
    params = fam.get("params")

    if params:
        t = partial(
            fam["t_norm"], **{k: kwargs[k] for k in params if k in kwargs}
        )
        s = partial(
            fam["t_conorm"], **{k: kwargs[k] for k in params if k in kwargs}
        )
        c = partial(
            fam["complement"], **{k: kwargs[k] for k in params if k in kwargs}
        )
    else:
        t = fam["t_norm"]
        s = fam["t_conorm"]
        c = fam["complement"]
    return {"t_norm": t, "t_conorm": s, "complement": c}


def min_t_norm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """Minimum t-norm operation between two fuzzy sets."""

    def min_membership(x):
        return np.minimum(a.mf(x), b.mf(x))

    return FuzzySet(
        name=f"MinTNorm({a.name}, {b.name})",
        membership_function=min_membership,
    )


def max_t_conorm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """Maximum t-conorm operation between two fuzzy sets."""

    def max_membership(x):
        return np.maximum(a.mf(x), b.mf(x))

    return FuzzySet(
        name=f"MaxTConorm({a.name}, {b.name})",
        membership_function=max_membership,
    )


def prod_t_norm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """Product t-norm operation between two fuzzy sets."""

    def prod_membership(x):
        return a.mf(x) * b.mf(x)

    return FuzzySet(
        name=f"ProdTNorm({a.name}, {b.name})",
        membership_function=prod_membership,
    )


def sum_t_conorm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """Sum t-conorm operation between two fuzzy sets."""

    def sum_membership(x):
        return a.mf(x) + b.mf(x) - a.mf(x) * b.mf(x)

    return FuzzySet(
        name=f"SumTConorm({a.name}, {b.name})",
        membership_function=sum_membership,
    )


def drastic_t_norm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """Drastic t-norm operation between two fuzzy sets."""

    def drastic_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        result = np.where(a_mf == 1, b_mf, np.where(b_mf == 1, a_mf, 0))
        return result

    return FuzzySet(
        name=f"DrasticTNorm({a.name}, {b.name})",
        membership_function=drastic_membership,
    )


def drastic_t_conorm(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """Drastic t-conorm operation between two fuzzy sets."""

    def drastic_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        result = np.where(a_mf == 0, b_mf, np.where(b_mf == 0, a_mf, 1))
        return result

    return FuzzySet(
        name=f"DrasticTConorm({a.name}, {b.name})",
        membership_function=drastic_membership,
    )


def dubois_prade_t_norm(a: FuzzySet, b: FuzzySet, p: float = 0.5) -> FuzzySet:
    """Dubois-Prade t-norm operation between two fuzzy sets."""
    assert (
        0 < p <= 1
    ), "Parameter p must be in (0, 1) for Dubois-Prade operators."

    def dp_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        denom = np.maximum(np.maximum(a_mf, b_mf), p)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, (a_mf * b_mf) / denom, 0.0)
        return result

    return FuzzySet(
        name=f"DuboisPradeTNorm({a.name}, {b.name})",
        membership_function=dp_membership,
    )


def dubois_prade_t_conorm(
    a: FuzzySet, b: FuzzySet, p: float = 0.5
) -> FuzzySet:
    """Dubois-Prade t-conorm operation between two fuzzy sets."""
    assert (
        0 < p <= 1
    ), "Parameter p must be in (0, 1) for Dubois-Prade operators."

    def dp_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        denom = np.maximum(np.maximum(1 - a_mf, 1 - b_mf), p)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, (1 - a_mf) * (1 - b_mf) / denom, 0.0)
        return result

    return FuzzySet(
        name=f"DuboisPradeTConorm({a.name}, {b.name})",
        membership_function=dp_membership,
    )


def yager_t_norm(a: FuzzySet, b: FuzzySet, p: float) -> FuzzySet:
    """Yager t-norm operation between two fuzzy sets."""
    assert p > 0, "Parameter p must be greater than 0 for Yager operators."

    def yager_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        result = np.maximum(
            0, 1 - (((1 - a_mf) ** p + (1 - b_mf) ** p) ** (1 / p))
        )
        return result

    return FuzzySet(
        name=f"YagerTNorm({a.name}, {b.name})",
        membership_function=yager_membership,
    )


def yager_t_conorm(a: FuzzySet, b: FuzzySet, p: float) -> FuzzySet:
    """Yager t-conorm operation between two fuzzy sets."""
    assert p > 0, "Parameter p must be greater than 0 for Yager operators."

    def yager_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        result = np.minimum(1, ((a_mf**p + b_mf**p) ** (1 / p)))
        return result

    return FuzzySet(
        name=f"YagerTConorm({a.name}, {b.name})",
        membership_function=yager_membership,
    )


def schweizer_sklar_t_norm(a: FuzzySet, b: FuzzySet, p: float) -> FuzzySet:
    """Schweizer-Sklar t-norm operation between two fuzzy sets."""
    assert (
        p > 0
    ), "Parameter p must be greater than 0 for Schweizer-Sklar operators."

    def schweizer_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        a_mf_ = (1 - a_mf) ** p
        b_mf_ = (1 - b_mf) ** p
        result = 1 - (a_mf_ + b_mf_ - a_mf_ * b_mf_) ** (1 / p)
        return result

    return FuzzySet(
        name=f"SchweizerTNorm({a.name}, {b.name})",
        membership_function=schweizer_membership,
    )


def schweizer_sklar_t_conorm(a: FuzzySet, b: FuzzySet, p: float) -> FuzzySet:
    """Schweizer-Sklar t-conorm operation between two fuzzy sets."""
    assert (
        p > 0
    ), "Parameter p must be greater than 0 for Schweizer-Sklar operators."

    def schweizer_membership(x):
        a_mf = a.mf(x)
        b_mf = b.mf(x)
        a_mf_ = a_mf**p
        b_mf_ = b_mf**p
        result = (a_mf_ + b_mf_ - a_mf_ * b_mf_) ** (1 / p)
        return result

    return FuzzySet(
        name=f"SchweizerTConorm({a.name}, {b.name})",
        membership_function=schweizer_membership,
    )


def complement_minus(a: FuzzySet, p: float = 0) -> FuzzySet:
    """Standard complement operation for a fuzzy set."""

    def neg_membership(x):
        return 1 - a.mf(x)

    return FuzzySet(
        name=f"Not({a.name})",
        membership_function=neg_membership,
    )


def complement_sugeno(a: FuzzySet, p: float) -> FuzzySet:
    """Sugeno complement operation fqor a fuzzy set."""
    assert p > -1, "Parameter p must be greater than -1 for Sugeno complement."

    def sugeno_membership(x):
        a_mf = a.mf(x)
        result = (1 - a_mf) / (1 + p * a_mf)
        return result

    return FuzzySet(
        name=f"SugenoNot({a.name})",
        membership_function=sugeno_membership,
    )


def complement_yager(a: FuzzySet, p: float) -> FuzzySet:
    """Yager complement operation for a fuzzy set."""
    assert p > 0, "Parameter p must be greater than 0 for Yager complement."

    def yager_membership(x):
        a_mf = a.mf(x)
        result = (1 - a_mf**p) ** (1 / p)
        return result

    return FuzzySet(
        name=f"YagerNot({a.name})",
        membership_function=yager_membership,
    )


if __name__ == "__main__":
    from bioclas.fuzzylogic.mem_functions import trimf, trapmf
    from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter
    from bioclas.fuzzylogic.fuzzy_set import FuzzySet
    from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable

    velocidad = FuzzyVariable("Velocidad", (0, 120), 0.1)
    quieto = FuzzySet("Quieto", lambda x: trimf(x, 0, 0, 30))
    baja = FuzzySet("Baja", lambda x: trimf(x, 0, 30, 50))
    media = FuzzySet("Media", lambda x: trimf(x, 30, 50, 70))
    alta = FuzzySet("Alta", lambda x: trapmf(x, 50, 70, 120, 120))

    op_family = get_fuzzy_family("schweizer-sklar", p=3)
    t_norm = op_family["t_norm"]
    t_conorm = op_family["t_conorm"]

    velocidad.add_fuzzysets([quieto, baja, media, alta])

    plotter = FuzzyPlotter()

    media_and_alta = t_norm(media, alta)
    media_or_alta = t_conorm(media, alta)
    not_alta = op_family["complement"](alta)

    plotter.add_fuzzy_set(media)
    plotter.add_fuzzy_set(alta)
    plotter.add_fuzzy_set(media_and_alta)
    plotter.add_fuzzy_set(media_or_alta)
    plotter.add_fuzzy_set(not_alta)

    plotter.domain = velocidad.interval
    plotter.plot(step=velocidad.step, title="Fuzzy Sets for Velocidad")
