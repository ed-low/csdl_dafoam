from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

# x.add_system("opt", OPT, r"\text{Optimizer}")
# x.add_system("solver", SOLVER, r"\text{Newton}")
# x.add_system("D1", FUNC, "D_1")
# x.add_system("D2", FUNC, "D_2")
# x.add_system("F", FUNC, "F")
# x.add_system("G", FUNC, "G")


# x.add_system("opt", OPT, r"0, 5$\rightarrow$1:\\\text{modopt}")
x.add_system("opt", OPT, r"\shortstack{\text{modopt}}")
x.add_system("geo", FUNC, r"\shortstack{\text{lsdo\_geo}}")
x.add_system("idwarp", FUNC, r"\shortstack{\text{IDWarp}}")
x.add_system("solver", SOLVER, r"\shortstack{\text{DAFoamSolver}}")
x.add_system("functions", FUNC, r"\shortstack{\text{DAFoamFunctions}}")
x.add_system("obj", FUNC, r"\shortstack{\text{Objective}}")

x.connect("opt", "geo", r"x_{geometry}")
x.connect("geo", "idwarp", r"p_{S}")
x.connect("idwarp", "solver", r"p_{V}")
x.connect("solver", "functions", r"y")
x.connect("opt", "solver", r"x_{flight}")
x.connect("functions", "obj", r"L, D, etc.")
x.connect("opt", "obj", r"x_{flight}")

x.connect("obj", "opt", r"f")

x.add_input("opt","x_0")



# x.connect("opt", "D1", "x, z")
# x.connect("opt", "D2", "z")
# x.connect("opt", "F", "x, z")
# x.connect("solver", "D1", "y_2")
# x.connect("solver", "D2", "y_1")
# x.connect("D1", "solver", r"\mathcal{R}(y_1)")
# x.connect("solver", "F", "y_1, y_2")
# x.connect("D2", "solver", r"\mathcal{R}(y_2)")
# x.connect("solver", "G", "y_1, y_2")

# x.connect("F", "opt", "f")
# x.connect("G", "opt", "g")

# x.add_output("opt", "x^*, z^*", side=LEFT)
# x.add_output("D1", "y_1^*", side=LEFT)
# x.add_output("D2", "y_2^*", side=LEFT)
# x.add_output("F", "f^*", side=LEFT)
# x.add_output("G", "g^*", side=LEFT)

x.write("connection_visualization")
