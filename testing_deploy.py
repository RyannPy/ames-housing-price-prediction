from flask import Flask, request, jsonify
import joblib

# inisialisasi flask
app = Flask(__name__)

# muat model
joblib_model = joblib.load("models/gbr_model.joblib")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = joblib_model.predict(data)
    return jsonify({'prediction' : prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)