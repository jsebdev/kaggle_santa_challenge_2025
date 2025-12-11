# %%

import numpy as np
from scipy.optimize import fsolve

# %%

# %%

# simple example of equations
# sin(a) = 2*b + a
# tan(b) = -1

def equations(vars):
    a, b = vars
    eq1 = np.cos(a) - (2 * b + a)
    eq2 = np.tan(b) + 1
    return [eq1, eq2]

initial_guess = [0.5, 0.5]
solution = fsolve(equations, initial_guess)

# %%

print('>>>>> solve_2_trees_equations.py:38 "solution"')
print(solution)

a = solution[0]
b = solution[1]

print('>>>>> solve_2_trees_equations.py:46 "np.sin(a), 2*b + a"')
print(np.sin(a), 2*b + a)
print('>>>>> solve_2_trees_equations.py:48 "np.tan(b)"')
print(np.tan(b))

# %%

print('>>>>> solve_2_trees_equations.py:53 "np.pi / 2"')
print(np.pi / 2)

print('>>>>> solve_2_trees_equations.py:53 "np.pi / 4"')
print(np.pi / 4)
