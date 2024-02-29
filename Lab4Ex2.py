import pandas as pd
from sklearn.tree import DecisionTreeClassifier
X = pd.DataFrame({'A': [1, 1, 0, 0],
                  'B': [1, 0, 1, 0],
                  'C': [0, 1, 1, 1]})
Y = pd.Series([0, 1, 1, 0])
#1)
tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=3)

tree.fit(X, Y)


y_pred = tree.predict(X)

accuracy = (y_pred == Y).mean()
print("Accuracy on training data:", accuracy)

#2) No, there isn't.