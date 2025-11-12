from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter
from bioclas.fuzzylogic.fuzzy_set import FuzzySet

import numpy as np

class FuzzyVariable:
    """A class representing a fuzzy variable with associated fuzzy sets."""

    def __init__(
        self, name: str, interval: tuple[float, float]
    ):
        """Initialize the fuzzy variable.

        Args:
            name (str): The name of the fuzzy variable.
            interval (tuple[float, float]): The domain of the variable.

        """
        self.__name = name
        self.__interval = interval
        self.__fuzzysets = {}

    @property
    def name(self) -> str:
        return self.__name

    @property
    def interval(self) -> tuple[float, float]:
        return self.__interval

    def add_fuzzyset(self, fuzzyset: FuzzySet) -> None:
        self.__fuzzysets[fuzzyset.name] = fuzzyset

    def add_fuzzysets(self, fuzzysets: list[FuzzySet]) -> None:
        for fs in fuzzysets:
            self.add_fuzzyset(fs)

    def has_fuzzyset(self, name: str) -> bool:
        return name in self.__fuzzysets

    def get_fuzzyset(self, name: str) -> FuzzySet:
        return self.__fuzzysets.get(name)

    def fuzzyset_names(self) -> list[str]:
        return list(self.__fuzzysets.keys())

    def dof(self, fuzzyset_name: str, value: float) -> float:
        fuzzyset = self.__fuzzysets.get(fuzzyset_name)
        if fuzzyset is None:
            raise ValueError(
                f"Fuzzy set '{fuzzyset_name}' not found in variable '{self._name}'."
            )
        return fuzzyset.dof(value)
    
    def defuzzify(self, degrees: dict[str, float], method: str = "centroid", step: float = 0.1) -> float:
        """Defuzzify the fuzzy variable using the specified method.

        Args:
            degrees (dict[str, float]): A dictionary mapping fuzzy set names to their degrees of membership.
            method (str): The defuzzification method to use. Currently only "centroid" is supported.
            step (float): The step size for numerical integration (used in centroid method).

        Returns:
            float: The defuzzified crisp value.
        """
        if method != "centroid":
            raise ValueError(f"Defuzzification method '{method}' not supported.")
        
        a, b = self.__interval

        numerator = 0.0
        denominator = 0.0

        mu = lambda x: np.zeros_like(x)

        

        # Build the aggregated membership function
        for fs_name, degree in degrees.items():
            if fs_name not in self.__fuzzysets:
                raise ValueError(f"Fuzzy set '{fs_name}' not found in variable '{self.__name}'.")
            if degree < 0.0 or degree > 1.0:
                raise ValueError(f"Degree of membership for fuzzy set '{fs_name}' must be in [0, 1].")
            fs = self.__fuzzysets[fs_name]

            mu = lambda x, mu=mu, fs=fs, degree=degree: np.maximum(mu(x), np.minimum(fs.mf(x), degree))

        x = np.arange(a, b, step)

        # plotter = FuzzyPlotter()
        # fss = FuzzySet("dummy", mu)  # Dummy initialization
        # plotter.add_fuzzy_set(fss)
        # plotter.domain = self.__interval
        # plotter.plot(step=step, title=f"Aggregated MF for '{self.__name}'", xlabel="x", ylabel="Membership Degree")

        numerator = np.sum(x * mu(x)) * step
        denominator = np.sum(mu(x)) * step

        if denominator == 0.0:
            raise ValueError("Denominator in defuzzification is zero.")
        
        # print(numerator, denominator)

        return numerator / denominator

class FuzzyVariableQualitative(FuzzyVariable):
    """A class representing a qualitative fuzzy variable."""

    def __init__(self, name: str, interval: tuple[int, int, int]):
        """Initialize the qualitative fuzzy variable.

        Args:
            name (str): The name of the fuzzy variable.
            interval (tuple[float, float]): The interval for the qualitative fuzzy variable.

        """
        super().__init__(name, interval)
        self.__colors = {}

    def add_color_fuzzyset(self, fuzzyset, color: tuple[int, int, int]) -> None:
        self.__colors[fuzzyset.name] = color
        super().add_fuzzyset(fuzzyset)

    @property
    def colors(self) -> dict[str, tuple[int, int, int]]:
        return self.__colors

    def defuzzify_color(self, degrees: dict[str, float]) -> tuple[int, int, int]:
        """Defuzzify the qualitative fuzzy variable to obtain a color.

        Args:
            degrees (dict[str, float]): A dictionary mapping fuzzy set names to their degrees of membership.

        Returns:
            tuple[int, int, int]: The defuzzified color as an RGB tuple.
        """
       
        # Defuzzification works by calculating a weighted average of the colors. Degrees are normalized first.
        total_degree = sum(degrees.values())
        if total_degree == 0.0:
            raise ValueError("Total degree of membership is zero; cannot defuzzify color.")
        r_defuzz = 0.0
        g_defuzz = 0.0
        b_defuzz = 0.0
        for fs_name, degree in degrees.items():
            if fs_name not in self.__colors:
                raise ValueError(f"Fuzzy set '{fs_name}' not found in variable '{self._name}'.")
            color = self.__colors[fs_name]
            normalized_degree = degree / total_degree
            r_defuzz += color[0] * normalized_degree
            g_defuzz += color[1] * normalized_degree
            b_defuzz += color[2] * normalized_degree

        return (int(r_defuzz), int(g_defuzz), int(b_defuzz))