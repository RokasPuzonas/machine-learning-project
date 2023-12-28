import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


data_set = pd.read_csv('counted-trimmed.csv')
X = data_set.iloc[:, :-1]
y = data_set.iloc[:, -1]

# normalization and log, and sqrt, pca, polyfit transformations did not help

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_train_encoded, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Optimal hyperparameter search
eta = 0.2 # 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1
depth = 9 # 3, 5, 7, 9, 11, 15, 18
rounds = 100 # 50, 100, 150, 200, 300
lamb = 1e-3 #1e-5, 1e-3, 0.01, 0.1
alpha = 1e-3 #1e-5, 1e-3, 0.01, 0.1
# subsample and colsample were considered, but default values 1 and 1 were optimal
# child weight was considered, but default value 1 was optimal
params = {
    'objective': 'multi:softmax',
    'num_class': len(set(y)),
    'max_depth': depth,
    'eta': eta,
    'eval_metric': 'mlogloss',
    'lambda': lamb,
    'alpha': alpha,
}

num_rounds = rounds
xgb_model = xgb.train(params, dtrain, num_rounds)

y_pred = xgb_model.predict(dtest)

y_pred = y_pred.astype(int)
y_test = y_test.astype(int)
accuracy = accuracy_score(y_test, y_pred)

best_array = [depth, eta, rounds, lamb, alpha, ]
y_pred = label_encoder.inverse_transform(y_pred)
y_test = label_encoder.inverse_transform(y_test)
report = classification_report(y_test, y_pred, output_dict=True)

print("Classification Report:")
print(report)
print("Best case")
print(f"Depth: {best_array[0]} Eta: {best_array[1]} Rounds: {best_array[2]} "
      f"Lambda: {best_array[3]} Alpha: {best_array[4]}  Accuracy: {accuracy}")

class_names = list(report.keys())[:-3] 
accuracy_values = [report[class_name]['precision'] for class_name in class_names]

conf = confusion_matrix(y_test, y_pred)

conf_data_frame = pd.DataFrame(conf, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_data_frame, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
