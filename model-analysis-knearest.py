import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

dataset = pd.read_csv('data/counted-trimmed.csv')
seed = 42

X = dataset.drop(columns=["target"])
Y = dataset["target"]

for column_name in X.columns:
    X[column_name] = 1/(X[column_name] + 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Best k=1
if False:
    for p in range(1, 10):
        # Create a KNeighborsClassifier with k (k neighbors)
        knn_classifier = KNeighborsClassifier(n_neighbors=p)

        # Fit the model to the training data
        knn_classifier.fit(X_train, Y_train)

        # Make predictions on the test data
        Y_pred = knn_classifier.predict(X_test)

        print(f"--- k={p} ---")
        # Evaluate the accuracy of the model
        accuracy = accuracy_score(Y_test, Y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        scores = cross_val_score(knn_classifier, X_train, Y_train, cv=5)
        print(f'Cross-validated Accuracy (5 fold): {scores.mean():.2f}')

# Best p=1
if False:
    for p in [1, 1.5, 2, 2.5, 3]:
        # Create a KNeighborsClassifier with k (k neighbors)
        knn_classifier = KNeighborsClassifier(n_neighbors=1, p=p)

        # Fit the model to the training data
        knn_classifier.fit(X_train, Y_train)

        # Make predictions on the test data
        Y_pred = knn_classifier.predict(X_test)

        print(f"--- p={p} ---")
        # Evaluate the accuracy of the model
        accuracy = accuracy_score(Y_test, Y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        scores = cross_val_score(knn_classifier, X_train, Y_train, cv=5)
        print(f'Cross-validated Accuracy (5 fold): {scores.mean():.2f}')

smote = SMOTE(random_state=seed)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# Create a KNeighborsClassifier with k (k neighbors)
knn_classifier = KNeighborsClassifier(n_neighbors=1, p=1)

# Fit the model to the training data
knn_classifier.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = knn_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')

scores = cross_val_score(knn_classifier, X_train, Y_train, cv=5)
print(f'Cross-validated Accuracy (5 fold): {scores.mean():.2f}')

labels = Y.unique()
confusion = confusion_matrix(Y_test, Y_pred, labels=labels)
diag_values = np.diag(confusion)
row_sums = np.sum(confusion, axis=1)
percentage_accuracy = (diag_values / row_sums)
for i in range(len(labels)):
    print(f"{labels[i]:10} {percentage_accuracy[i]*100:.2f}")