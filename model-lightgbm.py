import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading
dataset = pd.read_csv('data/counted-trimmed.csv')
seed = 42

# Data preparation
X = dataset.drop(columns=["target"])
Y = dataset["target"]

# Feature transformation
for column_name in X.columns:
    X[column_name] = 1 / (X[column_name] + 1)

# Data splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Creating a LightGBM classifier with the best hyperparameters
lgb_classifier = lgb.LGBMClassifier(
    learning_rate=0.01,
    max_depth=20,
    n_estimators=500,
    num_leaves=50,
    random_state=seed
)

# Training the classifier on the training data
lgb_classifier.fit(X_train, Y_train)

# Making predictions on the test data
Y_pred = lgb_classifier.predict(X_test)

# Evaluating the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

# Confusion matrix
labels = Y.unique()
confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)

# Class-specific accuracy calculation
diag_values = np.diag(confusion_mat)
row_sums = np.sum(confusion_mat, axis=1)
percentage_accuracy = diag_values / row_sums

# Bar graph for class-specific accuracies
if False:
    plt.figure(figsize=(10, 6))
    plt.bar(labels, percentage_accuracy * 100, color='skyblue')
    plt.xlabel('Class Label')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-specific Accuracy')
    plt.ylim(0, 100)
    plt.xticks(labels)
    plt.show()

# Heatmap for confusion matrix
if False:
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

# Printing class-specific accuracy
for i, label in enumerate(labels):
    print(f"Accuracy for {label}: {percentage_accuracy[i] * 100:.2f}%")
