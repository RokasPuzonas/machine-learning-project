import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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
rf_classifier = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    rf_classifier,
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

scores = cross_val_score(rf_classifier, X_train, Y_train, cv=5)
print(f'Cross-validated Accuracy (5 fold): {scores.mean():.2f}')

labels = Y.unique()
confusion = confusion_matrix(Y_test, Y_pred, labels=labels)
diag_values = np.diag(confusion)
row_sums = np.sum(confusion, axis=1)
percentage_accuracy = (diag_values / row_sums)
for i in range(len(labels)):
    print(f"{labels[i]:10} {percentage_accuracy[i]*100:.2f}")