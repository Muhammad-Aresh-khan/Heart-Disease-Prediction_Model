import utils.session
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#we will open our saved models and scaler here

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        input_values = [float(request.form[field]) for field in [
            'age', 'gender', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]

        # Scale and predict
        scaled_input = scaler.transform([input_values])
        prediction = model.predict(scaled_input)[0]

        result = "Likely cardiovascular disease — early intervention may be beneficial." if prediction == 1 else "Unlikely presence of cardiovascular disease — maintain a healthy lifestyle."
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
