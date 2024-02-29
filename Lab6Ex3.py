import numpy as np
import pandas as pd

class BernoulliJB:
    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = class_counts / len(y)

        self.feature_probs = {}
        for feature in X.columns:
            feature_prob = {}
            for value in X[feature].unique():
                feature_prob[value] = {}
                for label in self.classes:
                    subset = X[X[feature] == value]
                    feature_prob[value][label] = len(subset[subset[y.name] == label]) / class_counts[label]
            self.feature_probs[feature] = feature_prob

    def predict_proba(self, X):
        probabilities = np.zeros((len(X), len(self.classes)))

        for i, (_, row) in enumerate(X.iterrows()):
            for j, label in enumerate(self.classes):
                prob = np.log(self.class_probs[j])
                for feature in X.columns:
                    feature_prob = self.feature_probs[feature].get(row[feature], {}).get(label, 0)
                    prob += np.log(feature_prob)
                probabilities[i, j] = prob

        return np.exp(probabilities) / np.exp(probabilities).sum(axis=1, keepdims=True)

# Example usage:
d = pd.DataFrame({'X1': [0, 0, 1, 1, 0, 0, 1, 1],
                  'X2': [0, 0, 0, 0, 1, 1, 1, 1],
                  'C': [2, 18, 4, 1, 4, 1, 2, 18],
                  'Y': [0, 1, 0, 1, 0, 1, 0, 1]})
X = d[['X1', 'X2', 'Y']]  # Include 'Y' in the feature set
y = d['Y']

model = BernoulliJB()
model.fit(X, y)

# Predict probabilities for new data
new_data = pd.DataFrame({'X1': [0, 1],
                         'X2': [0, 1]})
proba = model.predict_proba(new_data)
print("1)")
print(proba)


#2)
print("2)\nAlgoritmul Joint Bayes estimează probabilități pentru fiecare clasă în problema de clasificare.\nNumărul de probabilități estimate este egal cu numărul de clase.")

#3)
# Instance with X1 = 0, X2 = 0
instance = pd.DataFrame({'X1': [0],
                          'X2': [0],})

# Probability estimates
proba_instance = model.predict_proba(instance)
print("3)")
print(proba_instance)