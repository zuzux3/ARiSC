from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
import logging

model = pickle.load(open('random_forest_regressor_model.pkl', 'rb'))

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug('Received a request')
    data = request.get_json(force=True)
    logging.debug(f'Data Received {data}')
    modelName = 'Random Forest Regressor'
    features = np.array(data['features'])
    
    prediction = model.predict([features])
    logging.debug(f'Prediction {prediction[0]}')
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True, port=8080)