from abc import ABC, abstractmethod

import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet

class FuzzyOperationError(Exception):
    """Custom exception for fuzzy operation errors."""
    def __init__(self, message: str):
        super().__init__(message)

class FuzzyOperationFactory:
    """Factory class for creating fuzzy logic operation sets (classes that inherit FuzzyOperationsSet)."""
    __registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a fuzzy operation set class with a given name."""
        def decorator(fuzzy_ops_class):
            cls.__registry[name] = fuzzy_ops_class
            return fuzzy_ops_class
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> 'FuzzyOperationsSet':
        """Create an instance of a registered fuzzy operation set class.

        Args:
            name (str): The name of the registered fuzzy operation set class.
            **kwargs: Additional parameters to pass to the class constructor.

        """
        if name not in cls.__registry:
            raise FuzzyOperationError(f"Fuzzy operation set '{name}' is not registered. Available sets: {list(cls.__registry.keys())}")
        fuzzy_ops_class = cls.__registry[name]
        return fuzzy_ops_class(**kwargs)

class FuzzyOperationsSet(ABC):
    """Abstract base class for fuzzy logic operations."""

    @abstractmethod
    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        raise NotImplementedError("t_norm method must be implemented by subclasses.")

    @abstractmethod
    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        raise NotImplementedError("t_conorm method must be implemented by subclasses.")

    @abstractmethod
    def complement(self, a: FuzzySet) -> FuzzySet:
        raise NotImplementedError("complement method must be implemented by subclasses.")

@FuzzyOperationFactory.register("min-max")
class MinMaxFuzzyOpsSet(FuzzyOperationsSet):
    """Fuzzy operations set using Min-Max methods."""
    def __init__(self):
        pass

    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return min_t_norm(a, b)

    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return max_t_conorm(a, b)

    def complement(self, a: FuzzySet) -> FuzzySet:
        return complement_minus(a)

@FuzzyOperationFactory.register("algebraic")
class AlgebraicFuzzyOpsSet(FuzzyOperationsSet):
    """Fuzzy operations set using Algebraic methods."""
    def __init__(self):
        pass

    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return prod_t_norm(a, b)

    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return sum_t_conorm(a, b)

    def complement(self, a: FuzzySet) -> FuzzySet:
        return complement_minus(a)

@FuzzyOperationFactory.register("drastic")   
class DrasticFuzzyOpsSet(FuzzyOperationsSet):
    """Fuzzy operations set using Drastic methods."""
    def __init__(self):
        pass

    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return drastic_t_norm(a, b)

    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return drastic_t_conorm(a, b)

    def complement(self, a: FuzzySet) -> FuzzySet:
        return complement_minus(a)

@FuzzyOperationFactory.register("dubois-prade")
class DuboisPradeFuzzyOpsSet(FuzzyOperationsSet):
    """Fuzzy operations set using Dubois-Prade methods."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return dubois_prade_t_norm(a, b, p=self.p)

    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return dubois_prade_t_conorm(a, b, p=self.p)

    def complement(self, a: FuzzySet) -> FuzzySet:
        return complement_minus(a)

@FuzzyOperationFactory.register("yager")
class YagerFuzzyOpsSet(FuzzyOperationsSet):
    """Fuzzy operations set using Yager methods."""
    def __init__(self, p: float):
        self.p = p

    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return yager_t_norm(a, b, p=self.p)

    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return yager_t_conorm(a, b, p=self.p)

    def complement(self, a: FuzzySet) -> FuzzySet:
        return complement_yager(a, p=self.p)

@FuzzyOperationFactory.register("schweizer-sklar")
class SchweizerSklarFuzzyOpsSet(FuzzyOperationsSet):
    """Fuzzy operations set using Schweizer-Sklar methods."""
    def __init__(self, p: float):
        self.p = p

    def t_norm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return schweizer_sklar_t_norm(a, b, p=self.p)

    def t_conorm(self, a: FuzzySet, b: FuzzySet) -> FuzzySet:
        return schweizer_sklar_t_conorm(a, b, p=self.p)

    def complement(self, a: FuzzySet) -> FuzzySet:
        return complement_minus(a)


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
            result = 1-np.where(denom > 0, (1 - a_mf) * (1 - b_mf) / denom, 0.0)
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
