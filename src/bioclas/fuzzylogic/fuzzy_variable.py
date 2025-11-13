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
    
    def plotter(self) -> FuzzyPlotter:
        plotter = FuzzyPlotter()
        plotter.add_fuzzy_variable(self)
        plotter.domain = self.__interval
        return plotter

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
    
    def defuzzify(self, degrees: dict[str, float], method: str = "centroid", imode: str = "mandami", step: float = 0.1) -> float:
        """Defuzzify the fuzzy variable using the specified method.

        Args:
            degrees (dict[str, float]): A dictionary mapping fuzzy set names to their degrees of fulfillment.
            method (str): The defuzzification method to use. Currently "centroid" and "averageMax" is supported.
            step (float): The step size for numerical integration (used in centroid method).

        Returns:
            float: The defuzzified crisp value.
        """
        a, b = self.__interval
        if imode not in ["mandami", "larsen"]:
            raise ValueError(f"Unsupported fuzzy inference mode: '{imode}'. Choose 'mandami' or 'larsen'.")
        tnorm = np.minimum if imode == "mandami" else lambda x, y: x * y
        tconorm = np.maximum if imode == "mandami" else lambda x, y: x + y - x * y
        x = np.arange(a, b, step)

        mu_x = np.zeros_like(x)

        # Build the aggregated membership function
        for fs_name, degree in degrees.items():
            if fs_name not in self.__fuzzysets:
                raise ValueError(f"Fuzzy set '{fs_name}' not found in variable '{self.__name}'.")
            if degree < 0.0 or degree > 1.0:
                raise ValueError(f"Degree of membership for fuzzy set '{fs_name}' must be in [0, 1].")
            fs = self.__fuzzysets[fs_name]
            mu_fs = tnorm(fs.mf(x), degree)
            mu_x = tconorm(mu_x, mu_fs)

        if method == "centroid":
            numerator = np.sum(x * mu_x) * step
            denominator = np.sum(mu_x) * step

            # from matplotlib import pyplot as plt
            # plt.figure()
            # plt.plot(x, mu_x, label="Aggregated MF")
            # plt.title(f"Aggregated Membership Function for Variable '{self.__name}'")
            # plt.ylim((0,1))
            # plt.xlabel("Universe of Discourse")
            # plt.ylabel("Membership Degree")
            # plt.legend()
            # plt.grid()
            # plt.show()

            if denominator == 0.0:
                raise ValueError("Denominator in defuzzification is zero.")
            return numerator / denominator
        elif method == "averageMax":
            max_mu = np.max(mu_x)
            x_max = x[mu_x > max_mu-0.1]
            if len(x_max) == 0:
                raise ValueError("No maximum found in membership function for averageMax defuzzification.")
            return np.mean(x_max)
        else:
            raise ValueError(f"Unsupported defuzzification method: '{method}'. Choose 'centroid' or 'averageMax'.")

    
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