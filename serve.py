"""
serve.py — Flask web service for hotel-cancellation prediction (cap. 5.2.3).

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict -> {"cancellation_probability": ..., "is_canceled": ...}

Run:
    python serve.py
    # then test:
    curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d @sample_request.json
"""

import pickle
from flask import Flask, request, jsonify

from predict import load_model, predict_one

app = Flask('hotel-cancellation')


# Load the model once at startup
MODEL_DATA = load_model('model.bin')


@app.route('/health', methods=['GET'])
def health():
    """Simple liveness probe."""
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    """Score a single booking.

    Expects JSON with booking fields. Returns:
    {
        "cancellation_probability": <float in [0,1]>,
        "is_canceled": <bool>
    }
    """
    booking = request.get_json(force=True)
    proba = predict_one(
        booking,
        MODEL_DATA['dv'],
        MODEL_DATA['model'],
        MODEL_DATA['features'],
    )
    result = {
        'cancellation_probability': float(proba),
        'is_canceled': bool(proba >= 0.5),
    }
    return jsonify(result)


if __name__ == '__main__':
    print(f"Loaded model: {MODEL_DATA['model_name']}  "
          f"(val AUC={MODEL_DATA['metrics']['val_auc']:.4f}, "
          f"test AUC={MODEL_DATA['metrics']['test_auc']:.4f})")
    app.run(debug=False, host='0.0.0.0', port=9696)
