from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from statistics import mean

x_red = norm.rvs(0, 1, 100, random_state=1)
y_red = norm.rvs(0, 1, 100, random_state=2)
x_green = norm.rvs(1, 1, 100, random_state=3)
y_green = norm.rvs(1, 1, 100, random_state=4)
d = pd.DataFrame({
    'X1': np.concatenate([x_red, x_green]),
    'X2': np.concatenate([y_red, y_green]),
    'Y': [1] * 100 + [0] * 100
})
X, Y = d[['X1', 'X2']], d['Y']

# 1
c = ['white' if l == 0 else 'black' for l in Y]
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_axisbelow(True)
plt.scatter(X['X1'], X['X2'], color=c, edgecolor='k')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(linestyle='dashed')
plt.show()

# 2
neighbors = np.arange(1, 16)
train_error = np.empty(len(neighbors))
cvloo_error = np.empty(len(neighbors))
loo = LeaveOneOut()
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, Y)
    accuracy = knn.score(X, Y)
    train_error[i] = 1 - accuracy
    cv_score = cross_val_score(knn, X, Y, cv=loo)
    cv_mean = mean(cv_score)
    cvloo_error[i] = 1 - cv_mean

plt.title('K Neighbors')
plt.plot(neighbors, train_error, label='Training Error')
plt.plot(neighbors, cvloo_error, label='CVLOO Error')
plt.legend()
plt.xlabel('No. Neighbors')
plt.ylabel('Error')
plt.show()

#3 best value for K is 10 or 14