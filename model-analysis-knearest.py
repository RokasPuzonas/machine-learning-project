import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
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

smote = SMOTE(random_state=seed)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# Create a KNeighborsClassifier with k (k neighbors)
knn_classifier = KNeighborsClassifier(n_neighbors=1, p=1)

# Define hyperparameter grid
param_grid = {
    'n_neighbors': [1, 5, 10, 15, 20, 25, 30],
    'p': [1, 2, 3, 4, 5],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    knn_classifier,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1,  # Use all available cores for parallel processing
)

# Perform grid search on the training data
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

scores = cross_val_score(knn_classifier, X_train, Y_train, cv=5)
print(f'Cross-validated Accuracy (5 fold): {scores.mean():.2f}')

labels = Y.unique()
confusion = confusion_matrix(Y_test, Y_pred, labels=labels)
diag_values = np.diag(confusion)
row_sums = np.sum(confusion, axis=1)
percentage_accuracy = (diag_values / row_sums)
for i in range(len(labels)):
    print(f"{labels[i]:10} {percentage_accuracy[i]*100:.2f}")