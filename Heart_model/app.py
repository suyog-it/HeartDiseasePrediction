from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model (ensure model.pkl exists in the same folder)
with open("heart_disease.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Expecting JSON from frontend
        # Extract features in the same order the model was trained
        features = np.array([
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]).reshape(1, -1)

        # Get prediction and probability
        prediction = model.predict(features)[0]
        try:
            confidence = float(np.max(model.predict_proba(features)))
        except:
            confidence = 0.85  # fallback if model has no predict_proba

        result = {
            "prediction": int(prediction),
            "confidence": confidence
        }
        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
