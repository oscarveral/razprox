class FuzzySet:
    """
    A class representing a fuzzy set with a name and a membership function.
    Attributes:
        name (str): The name of the fuzzy set.
        membership_function: The membership function defining the fuzzy set. It should be a callable that takes a value and returns its membership degree.
        domain: The domain over which the fuzzy set is defined (optional).
    Methods:
        get_membership_degree(value): Returns the membership degree of a given value.
    """
    def __init__(self, name: str, membership_function: callable, domain=None):
        assert(name is not None), "Name cannot be None"
        assert(membership_function is not None), "Membership function cannot be None"
        self._name = name
        self._membership_function = membership_function
        self._domain = domain

    def get_membership_degree(self, value) -> float:
        return self._membership_function(value)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def support(self):
        return self._membership_function.support

    def __repr__(self):
        return f"FuzzySet(name={self.name}, membership_function={self.membership_function})"