import pytest

SRC = ".."
if SRC not in __import__("sys").path:
    __import__("sys").path.insert(0, SRC)

from fuzzylogic.FuzzySet import FuzzySet
import fuzzylogic as fuzzy


class TestFuzzySet:
    def test_initialization(self):
        name = "TestSet"
        membership_function = fuzzy.trimf(0, 5, 10)
        fuzzy_set = FuzzySet(name, membership_function)

        assert fuzzy_set.name == name
        assert fuzzy_set._membership_function == membership_function

    def test_initialization_with_none_name(self):
        membership_function = fuzzy.trimf(0, 5, 10)
        with pytest.raises(AssertionError) as excinfo:
            FuzzySet(None, membership_function)
        assert "Name cannot be None" in str(excinfo.value)

    def test_initialization_with_none_membership_function(self):
        name = "TestSet"
        with pytest.raises(AssertionError) as excinfo:
            FuzzySet(name, None)
        assert "Membership function cannot be None" in str(excinfo.value)

    def test_get_membership_degree(self):
        name = "TestSet"
        membership_function = fuzzy.sets.Triangular(0, 5, 10)
        fuzzy_set = FuzzySet(name, membership_function)

        assert fuzzy_set.get_membership_degree(5) == 1.0
        assert fuzzy_set.get_membership_degree(0) == 0.0
        assert fuzzy_set.get_membership_degree(10) == 0.0
        assert fuzzy_set.get_membership_degree(2.5) == 0.5

    def test_support_property(self):
        name = "TestSet"
        membership_function = fuzzy.sets.Triangular(0, 5, 10)
        fuzzy_set = FuzzySet(name, membership_function)

        assert fuzzy_set.support == (0, 10)

    def test_repr(self):
        name = "TestSet"
        membership_function = fuzzy.sets.Triangular(0, 5, 10)
        fuzzy_set = FuzzySet(name, membership_function)

        expected_repr = (
            f"FuzzySet(name={name}, membership_function={membership_function})"
        )
        assert repr(fuzzy_set) == expected_repr
