import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

# Loading the data
dataset = pd.read_csv('data/counted-trimmed.csv')
seed = 42

# Data preparation
X = dataset.drop(columns=["target"])
Y = dataset["target"]

# Transforming features
for column_name in X.columns:
    X[column_name] = 1 / (X[column_name] + 1)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Creating a LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(random_state=seed)

# Setting up the hyperparameter grid
param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 500],
    'max_depth': [-1, 10, 20],
}

# Setting up GridSearchCV
grid_search = GridSearchCV(
    lgb_classifier,
    param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Performing hyperparameter grid search
grid_search.fit(X_train, Y_train)

# Printing the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluating the best model on the test data
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

# Printing the confusion matrix
confusion_mat = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", confusion_mat)

# Calculating and printing accuracy percentage for each class
labels = Y.unique()
diag_values = np.diag(confusion_mat)
row_sums = np.sum(confusion_mat, axis=1)
percentage_accuracy = diag_values / row_sums
for i, label in enumerate(labels):
    print(f"{label}: {percentage_accuracy[i] * 100:.2f}%")