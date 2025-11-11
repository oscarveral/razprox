import numpy as np

class FuzzySet:
    """An abstract class representing a real, one-dimensional, fuzzy set with a name and a membership function."""

    EPS = 1e-6

    def __init__(self, name: str, membership_function: callable):
        """Initialize the fuzzy set with a name and a membership function.

        Args:
            name (str): The name of the fuzzy set.
            membership_function (callable): A function that takes a numpy array and returns a numpy array of membership values.
        """
        assert name is not None, "Name cannot be None"
        assert isinstance(name, str), "Name must be a string"
        assert (
            membership_function is not None
        ), "Membership function cannot be None"
        assert callable(
            membership_function
        ), "Membership function must be callable"
        self._name = name
        self._membership_function = membership_function

    @classmethod
    def singleton(cls, value: float, name: str = "real"):
        # TODO: discrete domain problem
        """Create a singleton fuzzy set.

        Args:
            name (str): The name of the fuzzy set.
            value (float): The value at which the membership is 1.

        Returns:
            FuzzySet: A singleton fuzzy set instance.
        """
        return cls(name, lambda x: np.where(x == value, 1.0, 0.0))

    @property
    def name(self) -> str:
        return self._name

    def mf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the membership function for real-valued inputs.

        Args:
            x (np.ndarray): The input values for the membership function.

        Returns:
            np.ndarray: The membership values for the given input.
        """
        return self._membership_function(x)

    def mf_interval(
        self, interval: tuple[float, float], step: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the membership function over a specified interval.

        Args:
            interval (tuple[float, float]): The interval over which to evaluate the membership function. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the evaluation. Defaults to 0.1.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the x values and the corresponding membership values.
        """
        assert isinstance(interval, tuple), "Interval must be a tuple."
        assert (
            len(interval) == 2
        ), "Interval must be a tuple with two elements."
        assert all(
            isinstance(i, (int, float)) for i in interval
        ), "Interval must contain numeric values."

        x = np.arange(interval[0], interval[1], step)
        return x, self._membership_function(x)

    def support(
        self, interval: tuple[float, float], step: float = 0.1
    ) -> np.ndarray:
        """Get the support of the fuzzy set, i.e., the set of points where the membership degree is strictly above cero

        Args:
            interval (tuple[float, float]): The interval over which to compute the support. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the support computation. Defaults to 0.1.

        Returns:
            np.ndarray: The support of the fuzzy set.
        """
        x = np.arange(interval[0], interval[1], step)
        membership_values = self.mf(x)
        return x[membership_values > FuzzySet.EPS]

    def kernel(
        self, interval: tuple[float, float], step: float = 0.1
    ) -> np.ndarray:
        """Get the kernel of the fuzzy set, i.e., the set of points where the membership degree is equal to one.

        Args:
            interval (tuple[float, float]): The interval over which to compute the kernel. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the kernel computation. Defaults to 0.1.

        Returns:
            np.ndarray: The kernel of the fuzzy set.
        """
        x = np.arange(interval[0], interval[1], step)
        membership_values = self.mf(x)
        return x[1.0 - membership_values < FuzzySet.EPS]

    def is_empty(
        self, interval: tuple[float, float], step: float = 0.1
    ) -> bool:
        """Check if the fuzzy set is empty over a given interval.

        Args:
            interval (tuple[float, float]): The interval over which to check emptiness. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the computation. Defaults to 0.1.

        Returns:
            bool: True if the fuzzy set is empty over the interval, False otherwise.
        """
        x = np.arange(interval[0], interval[1], step)
        membership_values = self.mf(x)
        return np.all(membership_values < FuzzySet.EPS)

    def height(
        self, interval: tuple[float, float], step: float = 0.1
    ) -> float:
        """Compute the height of the fuzzy set over a given interval.

        Args:
            interval (tuple[float, float]): The interval over which to compute the height. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the computation. Defaults to 0.1.

        Returns:
            float: The height of the fuzzy set.
        """
        x = np.arange(interval[0], interval[1], step)
        membership_values = self.mf(x)
        return float(np.max(membership_values))

    def is_normal(
        self, interval: tuple[float, float], step: float = 0.1
    ) -> bool:
        """Check if the fuzzy set is normal over a given interval.

        Args:
            interval (tuple[float, float]): The interval over which to check normality. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the computation. Defaults to 0.1.

        Returns:
            bool: True if the fuzzy set is normal over the interval, False otherwise.
        """
        x = np.arange(interval[0], interval[1], step)
        membership_values = self.mf(x)
        return np.any(membership_values >= 1.0 - FuzzySet.EPS)

    def alpha_cut(
        self, alpha: float, interval: tuple[float, float], step: float = 0.1
    ) -> np.ndarray:
        """Get the alpha-cut of the fuzzy set, i.e., the set of points where the membership degree is greater than or equal to alpha.

        Args:
            alpha (float): The alpha level for the cut.
            interval (tuple[float, float]): The interval over which to compute the alpha-cut. Even if step divides the interval evenly, the endpoint is not included.
            step (float): The step size for the alpha-cut computation. Defaults to 0.1.

        Returns:
            np.ndarray: The alpha-cut of the fuzzy set.FuzzyVariable
        """
        x = np.arange(interval[0], interval[1], step)
        membership_values = self.mf(x)
        return x[membership_values > alpha - FuzzySet.EPS]

    def dof(self, value: float) -> float:
        """Evaluate the degree of membership for a single real-valued input."""
        result = self.mf(np.array([value]))
        return float(result[0])

    def __repr__(self):
        return f"FuzzySet(name={self.name}, membership_function={self._membership_function})"

    def __str__(self):
        return f"FuzzySet: {self.name}"


if __name__ == "__main__":
    speed = FuzzySet("Speed", lambda x: np.clip((75 - x) / 50, 0, 1))
    support = speed.support((0, 100), step=1.0)
    kernel = speed.kernel((0, 100), step=1.0)
    print(f"Support of '{speed.name}': {support}")
    print(f"Kernel of '{speed.name}': {kernel}")

    from bioclas.fuzzylogic.fuzzy_plotter import FuzzyPlotter

    plotter = FuzzyPlotter()
    plotter.add_fuzzy_set(speed)
    plotter.domain = (0, 100)
    plotter.plot(
        title="Fuzzy Set Support and Kernel Example",
        xlabel="Speed",
        ylabel="Membership Degree",
    )
