# ❤️ Heart Disease Prediction System

A supervised machine learning project using **Logistic Regression** to predict heart disease based on medical features. Includes a **Flask web application** for real-time predictions.

---

## 📊 Project Overview

This project uses the **Cleveland Heart Disease dataset** to classify whether a patient is likely to have heart disease. The pipeline includes data preprocessing, model training, evaluation, and a web interface for predictions.

---

## 🧠 Learning Type

- ✅ **Supervised Learning:** Model trained on labeled data (features + target).
- ❌ No clustering or unsupervised methods are used.

---

## 📁 Dataset Info

- **Total Samples:** 303  
- **Target Column:** `target`  
  - `0` = No Heart Disease  
  - `1` = Has Heart Disease

---

## 🔍 Features

| Feature    | Description                                      |
|------------|--------------------------------------------------|
| age        | Age in years                                     |
| sex        | Gender (0 = Female, 1 = Male)                    |
| cp         | Chest pain type (0–3)                            |
| trestbps   | Resting blood pressure                           |
| chol       | Serum cholesterol in mg/dl                       |
| fbs        | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| restecg    | Resting electrocardiographic results (0–2)       |
| thalach    | Maximum heart rate achieved                      |
| exang      | Exercise-induced angina (0 = No, 1 = Yes)        |
| oldpeak    | ST depression induced by exercise                |
| slope      | Slope of the ST segment (0–2)                    |
| ca         | Number of major vessels (0–3)                    |
| thal       | Thalassemia (1 = fixed, 2 = normal, 3 = reversible) |
| target     | Heart disease (0 = No, 1 = Yes)                  |

---

## ⚙️ Model Training Pipeline

### 🔍 Model Used
- **Logistic Regression**: A simple yet effective classification algorithm used to predict the likelihood of heart disease based on input features.

### 1. Prepare Features and Labels
```python
X = df.drop('target', axis=1).values
y = df['target'].values

## ⚙️ Model Training Pipeline

### 1. Prepare Features and Labels
```python
X = df.drop('target', axis=1).values
y = df['target'].values
```

### 2. Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Standardize Features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Model Training
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5. Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Evaluation metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### 📈 Model Performance
✅ Train Accuracy: ~84%

✅ Test Accuracy: ~88%

**Evaluation metrics used:**
- Confusion Matrix
  
    ![image](https://github.com/user-attachments/assets/598c5ca9-6cd1-49ef-8d1e-66963c295fb4)
  

- Classification Report
  
    ![image](https://github.com/user-attachments/assets/6d59dc24-9f23-4f74-b143-cea864cfb030)


### 6. Web-Based Prediction with Flask
```bash
# Clone the repository
$ git clone https://github.com/yourusername/Heart-Disease-Prediction_Model.git
$ cd Heart-Disease-Prediction_Model

# Install required packages
$ pip install -r requirements.txt

# Run the Flask app
$ python app.py
```

Then open your browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)
### 7. Demo
Positive Result

![image](https://github.com/user-attachments/assets/d5166c1b-527e-4656-881d-176030ef285f)



Negative Result

![image](https://github.com/user-attachments/assets/06597d0d-017f-424b-aea2-74f6f30d138d)

