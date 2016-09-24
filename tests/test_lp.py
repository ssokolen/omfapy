import pandas as pd
import sys

sys.path.append('..')

# Local imports
import omfapy as omfa
from omfapy.model import chebyshev_center, analytic_center

# Testing using triangle 
G = pd.DataFrame({'x':[1, -1, 0], 'y':[1, 1, -1]})
h = pd.Series([2, 0, 0])

x1 = chebyshev_center(G, h)
print(x1)

x2 = analytic_center(G, h)
print(x2)
print(h - G.dot(x2))
