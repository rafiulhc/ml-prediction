from flask import Flask, request, jsonify
import torch
from model import RegressionModel
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = RegressionModel()
model.load_state_dict(torch.load("regression_model.pth"))
model.eval()

# Initialize the scaler (same as used during training)
scaler = StandardScaler()
scaler.mean_ = np.array([0.0])  # Replace with your training data mean
scaler.scale_ = np.array([1.0])  # Replace with your training data scale

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get input data
        input_data = np.array(data['input']).reshape(-1, 1)
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            predictions = model(input_tensor).numpy()
        return jsonify({'prediction': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
