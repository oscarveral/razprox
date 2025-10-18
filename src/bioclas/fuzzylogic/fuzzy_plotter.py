import matplotlib.pyplot as plt
import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet


class FuzzyPlotter:
    """A class for plotting fuzzy sets and membership functions."""

    def __init__(self):
        self._fsets = []
        self._domain = (0, 1)  # Default domain

    def add_fuzzy_set(self, fuzzy_set: FuzzySet) -> None:
        """Add a fuzzy set to the plotter."""
        self._fsets.append(fuzzy_set)

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
        num_points: int = 100,
        title: str = "Fuzzy Sets",
        xlabel: str = "Universe of Discourse",
        ylabel: str = "Membership Degree",
    ) -> None:
        """Plot all added fuzzy sets over the specified domain.

        Args:
            num_points (int): Number of points to use for plotting.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.

        Returns:
            None
        """
        x = np.linspace(self._domain[0], self._domain[1], num_points)
        plt.figure()

        for fset in self._fsets:
            y = fset.mf(x)
            plt.plot(x, y, label=fset.name)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(self._domain[0], self._domain[1])
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid()
        plt.show()
