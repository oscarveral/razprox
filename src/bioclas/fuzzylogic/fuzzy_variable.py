from bioclas.fuzzylogic.fuzzy_set import FuzzySet

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
        self._name = name
        self._interval = interval
        self._fuzzysets = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    def add_fuzzyset(self, fuzzyset: FuzzySet) -> None:
        self._fuzzysets[fuzzyset.name] = fuzzyset

    def add_fuzzysets(self, fuzzysets: list[FuzzySet]) -> None:
        for fs in fuzzysets:
            self.add_fuzzyset(fs)

    def get_fuzzyset(self, name: str) -> FuzzySet:
        return self._fuzzysets.get(name)

    def fuzzyset_names(self) -> list[str]:
        return list(self._fuzzysets.keys())

    def dof(self, fuzzyset_name: str, value: float) -> float:
        fuzzyset = self._fuzzysets.get(fuzzyset_name)
        if fuzzyset is None:
            raise ValueError(
                f"Fuzzy set '{fuzzyset_name}' not found in variable '{self._name}'."
            )
        return fuzzyset.dof(value)
