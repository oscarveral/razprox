import matplotlib.pyplot as plt
import numpy as np

import bioclas.fuzzylogic.mem_functions


def plotting(x, y, title="Membership Function"):
    plt.plot(x, y)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    x = np.linspace(0, 10, 1001)
    y = bioclas.fuzzylogic.mem_functions.trimf(x, 2, 5, 8)
    plotting(x, y, title="Triangular Membership Function")

    y = bioclas.fuzzylogic.mem_functions.trimf(x, 2, 5, 5)
    plotting(x, y, title="Triangular Membership Function Degenerate Case")

    y = bioclas.fuzzylogic.mem_functions.trapmf(x, 3, 5, 7, 9)
    plotting(x, y, title="Trapezoidal Membership Function")

    y = y*y
    plotting(x, y, title="Trapezoidal Membership Function Squared")

    y = bioclas.fuzzylogic.mem_functions.trapmf(x, 5, 5, 7, 9)
    plotting(x, y, title="Trapezoidal Membership Function Degenerate Case")

    y = bioclas.fuzzylogic.mem_functions.sigmf(x, 1, 5)
    plotting(x, y, title="Sigmoidal Membership Function")

    y = bioclas.fuzzylogic.mem_functions.smf(x, 3, 7)
    plotting(x, y, title="S-shaped Membership Function")

    y = bioclas.fuzzylogic.mem_functions.smf(x, 5, 6)
    plotting(x, y, title="S-shaped Membership Function")

    y = bioclas.fuzzylogic.mem_functions.pimf(x, 2, 4, 6, 8)
    plotting(x, y, title="Pi-shaped Membership Function")

    y = bioclas.fuzzylogic.mem_functions.pimf(x, 3, 4, 6, 7)
    plotting(x, y, title="Pi-shaped Membership Function")


