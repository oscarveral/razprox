from bioclas.fuzzylogic import FuzzyOperationFactory, FuzzyOperationsSet, FuzzySet, FuzzyPlotter, mem_functions

import numpy as np

if __name__ == "__main__":

    # Create a fuzzy operation set using Min-Max operations
    minmax_ops = FuzzyOperationFactory.create("yager",p=0.9)

    # Create two simple fuzzy sets
    set_a = FuzzySet("A", lambda x: mem_functions.trimf(x, 2, 5, 7))
    set_b = FuzzySet("B", lambda x: mem_functions.trimf(x, 3, 6, 8))

    # Perform t-norm (AND) operation
    intersection_set = minmax_ops.t_norm(set_a, set_b)

    # Perform t-conorm (OR) operation
    union_set = minmax_ops.t_conorm(set_a, set_b)

    # Perform complement operation
    complement_set_a = minmax_ops.complement(set_a)

    # Test the results at a range of values
    x = np.linspace(0, 10, 1000)
    plotter = FuzzyPlotter()
    plotter.domain = (0, 10)
    plotter.add_fuzzy_set(set_a)
    plotter.add_fuzzy_set(set_b)
    plotter.add_fuzzy_set(intersection_set)
    plotter.add_fuzzy_set(union_set)
    # plotter.add_fuzzy_set(complement_set_a)
    plotter.plot(title="Fuzzy Operations using Yager", xlabel="x", ylabel="Membership Degree")

    