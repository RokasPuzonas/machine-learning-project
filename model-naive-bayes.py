import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns


data_set = pd.read_csv('counted-trimmed.csv')
X = data_set.iloc[:, :-1]
y = data_set.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=chi2, k=60)
X_train_best = selector.fit_transform(X_train, y_train)
X_test_best = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)

X_train_selected = X_train.iloc[:, selected_indices]
X_test_selected = X_test.iloc[:, selected_indices]

fit_prior_param = [True]  # True, False
alpha_param = [0.1]  # 0.1, 0.5, 1.0, 2.0

best_accuracy = 0
best_classification_report = ""
classification_rep = ""
y_pred = ""
for fit_prior in fit_prior_param:
    for alpha in alpha_param:
        nb_model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
        nb_model.fit(X_train_selected, y_train)

        y_pred = nb_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classification_report = classification_rep

print("Best Accuracy:", best_accuracy)
print("Classification Report:\n", best_classification_report)
class_names = list(classification_rep.keys())[:-3]
accuracy_values = [classification_rep[class_name]['precision'] for class_name in class_names]

cm = confusion_matrix(y_test, y_pred)

conf_data_frame = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_data_frame, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
class_names = list(classification_rep.keys())[:-3]
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8, 5))
plt.bar(class_names, class_accuracy, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy for Each Class')
plt.show()
