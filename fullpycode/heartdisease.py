import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/cleveland.csv', header=None)
df.columns = ['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
df['gender'] = df['gender'].map({0: 'female', 1: 'male'})
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'].fillna(df['thal'].mean(), inplace=True)
df['ca'].fillna(df['ca'].mean(), inplace=True)
df['gender'] = df['gender'].map({'female': 0, 'male': 1})

X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("\n===== Logistic Regression =====")
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Testing Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_test))

sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
 
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)   
