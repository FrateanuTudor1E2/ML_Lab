import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import numpy as np

exoplanets = pd.DataFrame([
  (205, 0),
  (205, 0),
  (260, 1),
  (380, 1),
  (205, 0),
  (260, 1),
  (260, 1),
  (380, 0),
  (380, 0)
], columns=['Temperature', 'Habitable'])

# 1)
X = exoplanets[['Temperature']]
Y = exoplanets['Habitable']

tree = DecisionTreeClassifier()
tree.fit(X, Y)

tree_rules = export_text(tree, feature_names=['Temperature'])
print("Decision Tree Rules:\n", tree_rules)

y_pred = tree.predict(X)
accuracy = (y_pred == Y).mean()
print("Training Accuracy:", accuracy)

# 2)
thresholds = []
for line in tree_rules.split('\n'):
    if 'Temperature' in line and '<=' in line:
        threshold = float(line.split(" <= ")[1].split()[0])
        thresholds.append(threshold)

manual_split_points = [205, 260, 380]

for i, (manual, extracted) in enumerate(zip(manual_split_points, thresholds)):
    print(f"Split Point {i + 1} - Manual: {manual}, Extracted: {extracted}")

# 3)
total_samples = len(exoplanets)
class_1_count = len(exoplanets[exoplanets['Habitable'] == 1])
class_0_count = total_samples - class_1_count

p_1 = class_1_count / total_samples
p_2 = class_0_count / total_samples

root_entropy = -p_1 * np.log2(p_1) - p_2 * np.log2(p_2)
print("Entropy of the Root Node:", root_entropy)