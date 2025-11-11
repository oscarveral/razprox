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

        self.__antecedents= {}
        self.__consequent = None


    def add_antecedent(
        self, variable: FuzzyVariable, fuzzyset_name: str
    ) -> None:
        """Add an antecedent to the fuzzy rule.
        
        Args:
            variable (FuzzyVariable): The fuzzy variable for the antecedent.
            fuzzyset_name (str): The name of the fuzzy set associated with the antecedent.

        Raises:
            ValueError: If the variable or fuzzy set name is invalid.
        """
        if variable is None or not isinstance(variable, FuzzyVariable):
            raise ValueError("Invalid fuzzy variable provided.")
        if fuzzyset_name is None or not isinstance(fuzzyset_name, str):
            raise ValueError("Invalid fuzzy set name provided.")
        if variable.get_fuzzyset(fuzzyset_name) is None:
            raise ValueError(
                f"Fuzzy set '{fuzzyset_name}' not found in variable '{variable.name}'."
            )
        self.__antecedents[variable.name] = (variable, fuzzyset_name)

    def set_consequent(
        self, variable: FuzzyVariable, fuzzyset_name: str
    ) -> None:
        """Set the consequent of the fuzzy rule.

        Args:
            variable (FuzzyVariable): The fuzzy variable for the consequent.
            fuzzyset_name (str): The name of the fuzzy set associated with the consequent.

        Raises:
            ValueError: If the variable or fuzzy set name is invalid.
        """
        if variable is None or not isinstance(variable, FuzzyVariable):
            raise ValueError("Invalid fuzzy variable provided.")
        if fuzzyset_name is None or not isinstance(fuzzyset_name, str):
            raise ValueError("Invalid fuzzy set name provided.")
        if variable.get_fuzzyset(fuzzyset_name) is None:
            raise ValueError(
                f"Fuzzy set '{fuzzyset_name}' not found in variable '{variable.name}'."
            )
        self.__consequent = (variable, fuzzyset_name)

    def eval(self, input_values: dict[str, float]) -> tuple[str, float]:
        """Evaluate the fuzzy rule given input values for the antecedents.

        Args:
            input_values (dict[str, float]): A dictionary mapping antecedent variable names to their input values.

        Returns:
            tuple[str, float]: A tuple containing the name of the consequent fuzzy set and its degree of membership.

        """
        if self.__consequent_var is None or not self.__antecedent_vars:
            raise ValueError("Fuzzy rule is not fully defined.")
        
        antecedent_result = 1.0
        for var in self.__antecedent_vars.keys():
            if var.name not in input_values:
                raise ValueError(f"Input value for '{var.name}' is missing.")
            value = input_values[var.name]
            fs_name = self.__antecedent_vars[var]
            degree = var.dof(fs_name, value)
            antecedent_result = min(antecedent_result, degree)

        return self.__consequent[1], antecedent_result
    
    @property
    def c_variable(self) -> FuzzyVariable:
        return self.__consequent[0] if self.__consequent else None

    @property
    def c_fuzzyset_name(self) -> str:
        return self.__consequent[1] if self.__consequent else None

    def __str__(self) -> str:
        antecedents_str = " AND ".join(
            [f"{var.name} IS {fs_name}" for var, fs_name in self.__antecedents]
        )
        consequent_str = (
            f"{self._consequent[0].name} IS {self._consequent[1]}"
            if self._consequent
            else "None"
        )
        return f"IF {antecedents_str} THEN {consequent_str}"
