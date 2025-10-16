import matplotlib.pyplot as plt
import numpy as np

import bioclas.fuzzylogic.mem_functions

def plotting(x ,y, title="Membership Function"):
    plt.plot(x,y)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    x = np.linspace(0, 10, 101)
    y = bioclas.fuzzylogic.mem_functions.trimf(x, 2, 5, 5)
    plotting(x, y, title="Triangular Membership Function")

    y2 = bioclas.fuzzylogic.mem_functions.trapmf(x, 3, 3, 7, 9)
    plotting(x, y2, title="Trapezoidal Membership Function")

    y3 = bioclas.fuzzylogic.mem_functions.sigmf(x, 1, 5)
    plotting(x, y3, title="Sigmoidal Membership Function")