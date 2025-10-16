import numpy as np

class FuzzySet():
    """An abstract class representing a fuzzy set with a name and a membership function."""

    def __init__(self, name: str, membership_function: callable, type: str = "real"):
        """Initialize the fuzzy set with a name and a membership function.
        
        Args:
            name (str): The name of the fuzzy set.
            membership_function (callable): A function that takes a numpy array and returns a numpy array of membership values.
            type (str): The type of the fuzzy set, default is "real". Possible values are "real" and "discrete".
        """ 
        assert(name is not None), "Name cannot be None"
        assert(isinstance(name, str)), "Name must be a string"
        assert(membership_function is not None), "Membership function cannot be None"
        assert(callable(membership_function)), "Membership function must be callable"
        self._name = name
        self._membership_function = membership_function
        self._mf_applier = self._mf_real if type == "real" else self._mf_discrete

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    def _mf_real(self, values: np.ndarray | float) -> np.ndarray:
        """Evaluate the membership function for real-valued inputs."""
        if isinstance(values, float) or isinstance(values, int):
            values = np.array([values])
        assert(isinstance(values, np.ndarray)), "Input must be a numpy array or a float"
        assert(values.ndim == 1), "Input array must be one-dimensional."
        assert(np.issubdtype(values.dtype, np.number)), "Input array must contain numeric values."
        return self._membership_function(values)

    def _mf_discrete(self, values: list) -> np.ndarray:
        """Evaluate the membership function for discrete inputs."""
        values = np.asarray(values)
        return self._membership_function(values)

    def mf(self, values: list | np.ndarray | float) -> np.ndarray:
        return self._mf_applier(values)

    def __repr__(self):
        return f"FuzzySet(name={self.name}, membership_function={self._membership_function})"
    
    def __str__(self):
        return f"FuzzySet: {self.name}"


if __name__ == "__main__":
    
    # Lambda should work with numpy arrays
    fuzzy_set = FuzzySet("Test Set", lambda x: np.clip(1 - np.abs(x - 10) / 10, 0, 1), type="real")
    v = np.linspace(0, 20, 21)
    print(fuzzy_set.mf(v))