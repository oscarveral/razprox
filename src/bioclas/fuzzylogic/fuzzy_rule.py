from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable


class FuzzyRule:
    """A class representing a fuzzy rule with antecedents and a consequent.

    A fuzzy rule consists of multiple antecedents (IF parts) and a single consequent (THEN part) in
    the form IF x1 is A1 and x2 is A2 ... THEN y is B, where xi are fuzzy variables, Ai are fuzzy sets
    associated with the antecedent variables, y is the consequent variable, and B is the fuzzy set
    associated with the consequent variable.
    """

    def __init__(self):
        """Initialize an empty fuzzy rule."""

        self._antecedents = []
        self._consequent = None

    def add_antecedent(
        self, variable: FuzzyVariable, fuzzyset_name: str
    ) -> None:
        """Add an antecedent to the fuzzy rule."""
        self._antecedents.append((variable, fuzzyset_name))

    def set_consequent(
        self, variable: FuzzyVariable, fuzzyset_name: str
    ) -> None:
        """Set the consequent of the fuzzy rule."""
        self._consequent = (variable, fuzzyset_name)

    def eval(self, input_values: list[float]) -> float | FuzzyVariable | str:
        assert self._consequent is not None, "Consequent is not set."
        assert self._antecedents, "No antecedents defined."
        assert len(input_values) == len(
            self._antecedents
        ), "Input values do not match number of antecedents."

        antecedent_result = 1.0
        for (var, fs_name), value in zip(self._antecedents, input_values):
            degree = var.dof(fs_name, value)
            antecedent_result = min(antecedent_result, degree)

        print(f"Antecedent evaluation result: {antecedent_result}")

        c_variable = self._consequent[0]
        c_fs_name = self._consequent[1]

        return (
            c_variable.dof(c_fs_name, antecedent_result),
            c_variable,
            c_fs_name,
        )

    def __str__(self) -> str:
        antecedents_str = " AND ".join(
            [f"{var.name} IS {fs_name}" for var, fs_name in self._antecedents]
        )
        consequent_str = (
            f"{self._consequent[0].name} IS {self._consequent[1]}"
            if self._consequent
            else "None"
        )
        return f"IF {antecedents_str} THEN {consequent_str}"
