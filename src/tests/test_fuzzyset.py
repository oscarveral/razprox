import pytest
import numpy as np

from bioclas.fuzzylogic.fuzzy_set import FuzzySet
from bioclas.fuzzylogic.mem_functions import trimf, trapmf, sigmf


class TestFuzzySet:
    world = np.linspace(0, 10, 101)
    mf = lambda _, x: trimf(x, 2, 5, 8)

    def test_initialization(self):
        fuzzy_set = FuzzySet("FS1", self.mf)
        assert fuzzy_set.name == "FS1"

    def test_initialization_with_none_name(self):
        with pytest.raises(AssertionError) as excinfo:
            FuzzySet(None, self.mf)
        assert "Name cannot be None" in str(excinfo.value)

    def test_initialization_with_none_membership_function(self):
        with pytest.raises(AssertionError) as excinfo:
            FuzzySet("FS2", None)
        assert "Membership function cannot be None" in str(excinfo.value)

    def test_get_membership_degree(self):
        fuzzy_set = FuzzySet("FS3", self.mf)

        assert fuzzy_set.mf(5) == np.array([1.0])
        assert fuzzy_set.mf(0) == np.array([0.0])
        assert fuzzy_set.mf(10) == np.array([0.0])
        assert fuzzy_set.mf(3.5) == np.array([0.5])

    def test_repr(self):
        fuzzy_set = FuzzySet("FS4", self.mf)

        expected_repr = f"FuzzySet(name=FS4, membership_function={self.mf})"
        assert repr(fuzzy_set) == expected_repr
