from bioclas.fuzzylogic.fuzzy_ops import FuzzyOperationsSet
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

        # Antecedents are stored as a dictionary mapping variable names to (variable, fuzzyset_name) tuples
        self.__antecedents= {}
        # Consequent is stored as a (variable, fuzzyset_name) tuple
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

    def eval(self, input_values: dict[str, float], mode="mandami") -> tuple[str, float]:
        """Evaluate the fuzzy rule given input values for the antecedents.

        Input values keys must contain all antecedent variable names.
        The agregation of the antecedents is done using the corresponding t-norm.

        Args:
            input_values (dict[str, float]): A dictionary mapping antecedent variable names to their input values.
            mode (str): The fuzzy inference mode. Currently "mandami" and "larsen" are supported.

        Raises:
            ValueError: If the fuzzy rule is not fully defined or if input values are missing.   

        Returns:
            tuple[FuzzyVariable, str, float]: A tuple containing 
            the fuzzy variable, the name of the consequent fuzzy set and its degree of membership.

        """
        if self.__consequent is None or not self.__antecedents:
            raise ValueError("Fuzzy rule is not fully defined.")
        
        antecedent_result = 1.0
        for var_n, (var, fs_name) in self.__antecedents.items():
            if var_n not in input_values:
                raise ValueError(f"Input value for '{var_n}' is missing.")
            value = input_values[var_n]
            degree = var.dof(fs_name, value)
            if mode == "mandami":
                antecedent_result = min(antecedent_result, degree)
            elif mode == "larsen":
                antecedent_result = antecedent_result * degree
            else:   
                raise ValueError(f"Unsupported fuzzy inference mode: '{mode}'. Choose 'mandami' or 'larsen'.")
            antecedent_result = min(antecedent_result, degree)

        return self.__consequent[0], self.__consequent[1], antecedent_result
    
    @property
    def c_variable(self) -> FuzzyVariable:
        return self.__consequent[0] if self.__consequent else None

    @property
    def c_fuzzyset_name(self) -> str:
        return self.__consequent[1] if self.__consequent else None

    def __str__(self) -> str:
        antecedents_str = " AND ".join(
            [f"{var_n} IS {fs_name}" for var_n, (var, fs_name) in self.__antecedents.items()]
        )
        consequent_str = (
            f"{self.__consequent[0].name} IS {self.__consequent[1]}"
            if self.__consequent
            else "None"
        )
        return f"IF {antecedents_str} THEN {consequent_str}"
