import matplotlib.pyplot as plt
import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet
from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable


class FuzzyPlotter:
    """A class for plotting fuzzy sets and membership functions."""

    def __init__(self):
        self._fvars = []
        self._fsets = []
        self._domain = (0, 1)  # Default domain

    def add_fuzzy_set(self, fuzzy_set: FuzzySet) -> None:
        """Add a fuzzy set to the plotter."""
        self._fsets.append(fuzzy_set)

    def add_fuzzy_sets(self, fuzzy_sets: list[FuzzySet]) -> None:
        """Add multiple fuzzy sets to the plotter."""
        self._fsets.extend(fuzzy_sets)

    def add_fuzzy_variable(self, fuzzy_var: FuzzyVariable) -> None:
        """Add a fuzzy variable to the plotter."""
        self._fvars.append(fuzzy_var)

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    @domain.setter
    def domain(self, interval: tuple[float, float]) -> None:
        """Set the domain for plotting.

        Args:
            interval (tuple[float, float]): A tuple specifying the (min, max) of the domain.

        Returns:
            None
        """
        assert isinstance(interval, tuple), "Domain must be a tuple."
        assert len(interval) == 2, "Domain must be a tuple with two elements."
        assert all(
            isinstance(i, (int, float)) for i in interval
        ), "Domain must contain numeric values."
        self._domain = interval

    def plot(
        self,
        step: float = 0.1,
        title: str = "Fuzzy Sets",
        xlabel: str = "Universe of Discourse",
        ylabel: str = "Membership Degree",
    ) -> None:
        """Plot all added fuzzy sets over the specified domain.

        Args:
            step (float): Step size for the x-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.

        Returns:
            None
        """
        x = np.arange(self._domain[0], self._domain[1], step)
        plt.figure()

        for fset in self._fsets:
            y = fset.mf(x)
            plt.plot(x, y, label=fset.name)

        for fvar in self._fvars:
            for fset_name in fvar.fuzzyset_names():
                fset = fvar.get_fuzzyset(fset_name)
                y = fset.mf(x)
                plt.plot(x, y, label=f"{fvar.name} - {fset.name}")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(self._domain[0], self._domain[1])
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()
        plt.show()
