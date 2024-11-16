from flask import Flask, request, jsonify
import torch
from model import SimpleModel

app = Flask(__name__)

# Load the model
model = SimpleModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = torch.tensor(data['input']).float()
        with torch.no_grad():
            output = model(input_data)
        return jsonify({'prediction': output.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
