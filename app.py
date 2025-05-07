from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
loaded_model = load_model("my_model.h5")

# Initialize the LabelEncoder and StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input JSON data
    input_data = request.get_json()

    # Ensure scalar values are converted to lists (to form a valid DataFrame)
    for key in input_data:
        if not isinstance(input_data[key], list):
            input_data[key] = [input_data[key]]

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Apply label encoding to categorical features
    input_df['name'] = label_encoder.fit_transform(input_df['name'])
    input_df['genre'] = label_encoder.fit_transform(input_df['genre'])
    input_df['artists'] = label_encoder.fit_transform(input_df['artists'])
    input_df['album'] = label_encoder.fit_transform(input_df['album'])

    # Ensure numerical columns are float
    input_df['popularity'] = input_df['popularity'].astype(float)

    # Scale the data
    X_scaled = scaler.fit_transform(input_df)

    # Predict
    y_pred_prob = loaded_model.predict(X_scaled)
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    # Return result
    return jsonify({
        'prediction': int(y_pred_class[0][0]),
        'confidence_level': float(y_pred_prob[0][0])
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
