from bioclas.fuzzylogic.fuzzy_variable import FuzzyVariable
from bioclas.fuzzylogic.fuzzy_rule import FuzzyRule

class FIS():
    """A class representing a Fuzzy Inference System (FIS).
    
    Every rule in the FIS should have the same antecedent variables (missing variables is allowed,
    in which case they are ignored for that rule) and the same consequent variable.
    """

    def __init__(self, antecedents: list[FuzzyVariable], consequent: FuzzyVariable) -> None:
        """Initialize the FIS with a name, empty variables and rules.
        
        Args:
            antecedents (list[FuzzyVariable]): A list of antecedent fuzzy variables.
            consequent (FuzzyVariable): The consequent fuzzy variable.
        """
        if not antecedents:
            raise ValueError("At least one antecedent variable must be provided.")
        self.__a_vars = {vn: v for vn, v in zip([v.name for v in antecedents], antecedents)}
        self.__consequent = consequent
        self.__rules = {}

    def add_rule(self, rule_name, antecedents: dict, consequent_fs_name: str) -> None:
        """Add a fuzzy rule to the FIS.
        
        Args:
            antecedents (dict): A dictionary mapping antecedent variable names to fuzzy set names.
            consequent_fs_name (str): The fuzzy set name for the consequent variable.
        """
        if not isinstance(antecedents, dict):
            raise ValueError("Antecedents must be a dictionary.")
        if not antecedents:
            raise ValueError("At least one antecedent must be provided.")
        if not consequent_fs_name:
            raise ValueError("A consequent must be provided.")
        if not isinstance(consequent_fs_name, str) or not self.__consequent.has_fuzzyset(consequent_fs_name):
            raise ValueError(f"Invalid consequent fuzzy set name provided. Given: '{consequent_fs_name}'")

        fuzzy_rule = FuzzyRule()
        for var_name, fs_name in antecedents.items():
            var = self.__a_vars.get(var_name, None)
            if var is None:
                raise ValueError(f"Antecedent variable '{var_name}' not found for rule '{rule_name}'.")
            else:
                fuzzy_rule.add_antecedent(var, fs_name)

        fuzzy_rule.set_consequent(self.__consequent, consequent_fs_name)
        self.__rules[rule_name] = fuzzy_rule

    @property
    def rules(self) -> list[FuzzyRule]:
        return self._rules
    
    @property
    def antecedent_vars(self) -> dict[str, FuzzyVariable]:
        return self.__a_vars
    
    @property
    def consequent(self) -> FuzzyVariable:
        return self.__consequent
    
    def eval(self, input_values: dict[str, float], mode: str = "mandami") -> dict[str, float]:
        """Evaluate the FIS given input values for the antecedent variables.

        The agregation of the outputs for the antecedent variables is done using the corresponding t-norm.
        The agregation of the outputs for the consequent variable is done using the corresponding t-conorm.

        This method works only on convex fuzzy sets.

        Example, if * denotes t-norm and + denotes t-conorm:

        IF x is A AND y is B THEN z is C -> dof1 = m(x,A) * m(y,B)
        IF u is D AND v is E THEN z is C -> dof2 = m(u,D) * m(v,E)
        Output dof for C = dof1 + dof2

        Args:
            input_values (dict[str, float]): A dictionary mapping antecedent variable names to their input values.

        Returns:
            dict[str, float]: A dictionary mapping consequent variable fuzzy set names to their output values. Aggregation for dof uses max operator.
        """
        output_values = {}
        for rule_n, rule in self.__rules.items():
            var, set_n, degree = rule.eval(input_values, mode=mode)
            if mode == "mandami":
                output_values[set_n] = max(degree, output_values.get(set_n, 0.0))
            elif mode == "larsen":
                x = output_values.get(set_n, 0.0)
                output_values[set_n] = x + degree - x * degree
            else:
                raise ValueError(f"Unsupported fuzzy inference mode: '{mode}'. Choose 'mandami' or 'larsen'.")
        return self.__consequent, output_values
