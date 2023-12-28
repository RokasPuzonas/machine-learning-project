import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/counted-trimmed.csv')
seed = 42

X = dataset.drop(columns=["target"])
Y = dataset["target"]

for column_name in X.columns:
    X[column_name] = 1/(X[column_name] + 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

smote = SMOTE(random_state=seed)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# Create a KNeighborsClassifier with k neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=1, p=1)

# Fit the model to the training data
knn_classifier.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = knn_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')

scores = cross_val_score(knn_classifier, X_train, Y_train, cv=10)
print(f'Cross-validated Accuracy (10 fold): {scores.mean():.2f}')

labels = Y.unique()
confusion = confusion_matrix(Y_test, Y_pred, labels=labels)
diag_values = np.diag(confusion)
row_sums = np.sum(confusion, axis=1)
percentage_accuracy = (diag_values / row_sums)
for i in range(len(labels)):
    print(f"{labels[i]:10} {percentage_accuracy[i]*100:.2f}")

if False:
    plt.bar(labels, percentage_accuracy)
    plt.title("Per class accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(top=1)
    plt.show()

sns.heatmap(confusion, fmt=".0f", annot=True, xticklabels=labels, yticklabels=labels)
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()