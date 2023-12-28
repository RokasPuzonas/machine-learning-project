import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

dataset = pd.read_csv('data/counted-trimmed.csv')
seed = 42

X = dataset.drop(columns=["target"])
Y = dataset["target"]

for column_name in X.columns:
    X[column_name] = 1/(X[column_name] + 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=150, random_state=seed)

# Train the classifier on the training data
rf_classifier.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

scores = cross_val_score(rf_classifier, X_train, Y_train, cv=5)
print(f'Cross-validated Accuracy (5 fold): {scores.mean():.2f}')

labels = Y.unique()
confusion = confusion_matrix(Y_test, Y_pred, labels=labels)
diag_values = np.diag(confusion)
row_sums = np.sum(confusion, axis=1)
percentage_accuracy = (diag_values / row_sums)
for i in range(len(labels)):
    print(f"{labels[i]:10} {percentage_accuracy[i]*100:.2f}")