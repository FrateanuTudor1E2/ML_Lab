from sklearn.datasets import load_iris
import pandas as pd
from sklearn import tree
import numpy as np
iris = load_iris()
d = pd.DataFrame(
    data = np.c_[iris['data'], pd.Categorical.from_codes(iris.target, iris.target_names)],
    columns= iris['feature_names'] + ['class'])

# Your code here

X = d.iloc[:,:4]
print(X)
y = d['class']
print(y)
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X,y)
X_pred = pd.DataFrame([
    (6.1,2.8,4.0,1.3)
])
print(dt.predict(X_pred))
print(dt.predict_proba(X_pred))