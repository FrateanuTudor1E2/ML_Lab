import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
import sklearn
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB


def apply_counts(df: pd.DataFrame, count_col: str):

    feats = [c for c in df.columns if c != count_col]
    return pd.concat([
        pd.DataFrame([list(r[feats])] * r[count_col], columns=feats)
        for i, r in df.iterrows()
    ], ignore_index=True)


def plot_decision_surface(clas, X, Y):
    h = .02
    cmap_light = ListedColormap(['lightblue', 'lightcoral'])
    cmap_bold = ListedColormap(['green','red'])

    x_min, x_max = X['X1'].min() - 1, X['X1'].max() + 1
    y_min, y_max = X['X2'].min() - 1, X['X2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clas.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')


    plt.scatter(X['X1'], X['X2'], c=Y, cmap=cmap_bold, s=20)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Classification")
    plt.show()

# EXERCISE 3
d = pd.DataFrame({'X1': [1, 2, 3, 3, 3, 4, 5, 5, 5],
                  'X2': [2, 3, 1, 2, 4, 4, 1, 2, 4],
                  'Y': [1, 1, 0, 0, 0, 0, 1, 1, 0]})
X, Y = d[['X1', 'X2']], d['Y']
c = ['green' if l == 0 else 'red' for l in Y]
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(X['X1'], X['X2'], color=c)
classifier = tree.DecisionTreeClassifier(criterion='entropy').fit(X, Y)
plot_decision_surface(classifier, X, Y)
plt.show()
X3 = []
X4 = []
Y2 = []
while len(Y2) < 1000:
    random1 = 10 * np.random.random_sample()
    random2 = 10 * np.random.random_sample()
    if random1 != random2:
        X3.append(random1)
        X4.append(random2)
    if random1 > random2:
        Y2.append(0)
    else:
        Y2.append(1)
d2 = pd.DataFrame({'X1': X3,
                   'X2': X4,
                   'Y': Y2})
Xs, Ys = d2[['X1', 'X2']], d2['Y']
c = ['green' if l == 0 else 'red' for l in Ys]
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(Xs['X1'], Xs['X2'], color=c)
classifier1 = tree.DecisionTreeClassifier(criterion='entropy').fit(Xs, Ys)
plot_decision_surface(classifier1, Xs, Ys)
plt.show()

# EXERCISE 5
d_grouped = pd.DataFrame({
    'X1': [0, 0, 1, 1, 0, 0, 1, 1],
    'X2': [0, 0, 0, 0, 1, 1, 1, 1],
    'C': [2, 18, 4, 1, 4, 1, 2, 18],
    'Y': [0, 1, 0, 1, 0, 1, 0, 1]})
d = apply_counts(d_grouped, 'C')
X = d[['X1', 'X2']]
Y = d['Y']
cl = BernoulliNB().fit(X, Y)
new_message = pd.DataFrame(
    [(0, 0)], columns=['X1', 'X2'])
print("I ", cl.predict(new_message))
print("II ", cl.predict_proba(new_message), " = 0.24")
print("III ", cl.class_log_prior_, " AND ", cl.feature_log_prob_, "Number of cells -> 6 ")

























































































































































































