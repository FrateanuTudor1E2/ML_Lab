import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d

d = pd.DataFrame({
    'X1': [2, 2, 3, 6, 8, 10, 10, 13, 14],
    'X2': [1, 8, 6, 10, 4, 6, 14, 13, 8],
    'Y': [0, 0, 0, 0, 0, 1, 1, 1, 1]
})
X, Y = d[['X1', 'X2']], d['Y']

# Correct this line => (10, 0),(6, 16)
boundary = pd.DataFrame([(9.8, 0), (7.3, 16)], columns=['x', 'y']) # corrected line

import matplotlib.pyplot as plt

c = ['white' if l == 0 else 'black' for l in Y]
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_axisbelow(True)
plt.scatter(X['X1'], X['X2'], color=c, edgecolor='k')
plt.xlim(0, 16)
plt.ylim(0, 16)
plt.plot(boundary['x'], boundary['y'])
plt.grid(linestyle='dashed')
plt.show()
