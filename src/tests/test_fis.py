from bioclas.fuzzylogic import FuzzyRule, FuzzySet, FuzzyVariable, FuzzyPlotter, FIS

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

    fis = FIS(antecedents=[var_e, var_De], consequent=var_Du)
    fis.add_rule(
        rule_name="rule1",
        antecedents={"e": "negativo"},
        consequent_fs_name="muy alta"
    )
    fis.add_rule(
        rule_name="rule2",
        antecedents={"e": "cero"},
        consequent_fs_name="media"
    )
    fis.add_rule(
        rule_name="rule3",
        antecedents={"e": "positivo"},
        consequent_fs_name="muy baja"
    )
    fis.add_rule(
        rule_name="rule4",
        antecedents={"e": "cero", "De": "positivo"},
        consequent_fs_name="baja"
    )
    fis.add_rule(
        rule_name="rule5",
        antecedents={"e": "cero", "De": "negativo"},
        consequent_fs_name="alta"
    )
    var, output = fis.eval({"e": 0.8, "De": 0.5})
    print("FIS output values:")
    for fs_name, value in output.items():
        print(f"  {fs_name}: {value}")

    defuzzified_value = var.defuzzify(output, method="centroid", step=0.001)
    print(f"Defuzzified output value for 'Du': {defuzzified_value}")