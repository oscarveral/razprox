from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable
from bioclas.fuzzylogic.fuzzy_rule import FuzzyRule

class FIS():
    """A class representing a Fuzzy Inference System (FIS)."""

    def __init__(self, antecedents: dict, consequent: dict):
        """Initialize the FIS with a name, empty variables and rules."""
        self.__a_vars = {}
        self.__c_vars = {}
        self.__rules = {}

    def add_rule(self, antecedents: dict, consequent: dict) -> None:
        """Add a fuzzy rule to the FIS.
        
        Args:
            antecedents (dict): A dictionary mapping antecedent variable names to fuzzy set names.
            consequent (dict): A dictionary mapping the consequent variable name to the fuzzy set name.
        """
        if not isinstance(antecedents, dict) or not isinstance(consequent, dict):
            raise ValueError("Antecedents and consequent must be dictionaries.")
        if not antecedents:
            raise ValueError("At least one antecedent must be provided.")
        if not consequent:
            raise ValueError("A consequent must be provided.")
        if len(consequent) != 1:
            raise ValueError("There must be exactly one consequent.")

        fuzzy_rule = FuzzyRule()
        for var_name, fs_name in antecedents.items():
            var = self.__a_vars.get(var_name)
            if var:
                fuzzy_rule.add_antecedent(var, fs_name)
            else:
                raise ValueError(f"Antecedent variable '{var_name}' not found.")
        for var_name, fs_name in consequent.items():
            var = self.__c_vars.get(var_name)
            if var:
                fuzzy_rule.set_consequent(var, fs_name)
        self.__rules.append(fuzzy_rule)

    @property
    def variables(self) -> dict[str, FuzzyVariable]:
        return self._variables

    @property
    def rules(self) -> list[FuzzyRule]:
        return self._rules
    
    def eval(self, input_values: dict[str, float]) -> dict[str, float]:
        """Evaluate the FIS given input values for the antecedent variables.

        Args:
            input_values (dict[str, float]): A dictionary mapping antecedent variable names to their input values.

        Returns:
            dict[str, float]: A dictionary mapping consequent variable fuzzy set names to their output values. Aggregation uses max operator.
        """
        output_values = {}
        for rule in self.__rules:
            set_n, degree = rule.eval(input_values)
            if degree > 0:
                output_values[set_n] = max(degree, output_values.get(set_n, 0.0))
        return output_values
