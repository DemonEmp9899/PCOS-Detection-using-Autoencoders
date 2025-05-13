from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allow CORS for all origins - simplifies development
CORS(app)

# Load your model
try:
    model = tf.keras.models.load_model("pcos_model.h5")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "success", "message": "API is working!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight requests
    if request.method == "OPTIONS":
        return "", 200
    
    # For standard POST requests
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Extract and validate the features from the request
        required_fields = ['weight', 'height', 'bmi', 'skinDarkening', 'hairGrowth', 
                          'weightGain', 'fastFood', 'pimples', 'hairLoss', 'cycle']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract features with proper type conversion
        try:
            input_data = [
                float(data['weight']),
                float(data['height']),
                float(data['bmi'] or 0),  # Use 0 if bmi is empty
                1 if data['skinDarkening'] else 0,
                1 if data['hairGrowth'] else 0,
                1 if data['weightGain'] else 0,
                1 if data['fastFood'] else 0,
                1 if data['pimples'] else 0,
                1 if data['hairLoss'] else 0,
                1 if data['cycle'] == 'Irregular' else 0,
            ]
            logger.info(f"Processed input data: {input_data}")
        except ValueError as e:
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Error processing input: {str(e)}"}), 400

        # Make prediction
        try:
            input_array = np.array([input_data])  # Shape (1, N)
            prediction = model.predict(input_array)[0][0]
            result = bool(prediction > 0.5)
            logger.info(f"Prediction: {prediction}, Result: {result}")
            
            return jsonify({"pcos_prediction": result})
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": f"Error making prediction: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)