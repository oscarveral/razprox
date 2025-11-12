from bioclas.fuzzylogic import FuzzyRule, FuzzySet, FuzzyVariable, FuzzyPlotter

if __name__ == "__main__":
    var_e = FuzzyVariable("e", (-20,20))
    var_e.add_fuzzysets([
        FuzzySet.trapezoidal(name="positivo", a=0, b=2, c=20, d=20),
        FuzzySet.triangular("cero", -2, 0, 2),
        FuzzySet.trapezoidal("negativo", -20, -20, -2, 0),
    ])

    var_De = FuzzyVariable("De", (-1,1))
    var_De.add_fuzzysets([
        FuzzySet.trapezoidal(name="positivo", a=0, b=0.5, c=1, d=1),
        FuzzySet.triangular("cero", -0.5, 0, 0.5),
        FuzzySet.trapezoidal("negativo", -1, -1, -0.5, 0),
    ])

    var_Du = FuzzyVariable("Du", (-40,40))
    var_Du.add_fuzzysets([
        FuzzySet.trapezoidal(name="muy alta", a=10, b=20, c=40, d=40),
        FuzzySet.triangular("alta", 0, 10, 20),
        FuzzySet.triangular("media", -2, 0, 2),
        FuzzySet.triangular("baja", -20, -10, 0),
        FuzzySet.trapezoidal("muy baja", -40, -40, -20, -10)
    ])

    rule = FuzzyRule()
    rule.add_antecedent(var_e, "positivo")
    rule.set_consequent(var_Du, "muy baja")

    _, _, degree = rule.eval({"e": 0.8, "De": 0.5, "Dx": 0.0})
    print(f"Degree of membership in the consequent fuzzy set: {degree}")