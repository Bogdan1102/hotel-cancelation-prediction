"""
predict.py — Load the model and make a prediction for one booking.

Used by serve.py; can also be run standalone for testing.

Usage (standalone):
    python predict.py
"""

import pickle
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


def load_model(model_path='model.bin'):
    """Load the DictVectorizer + model from a pickle file (cap. 5.1.2)."""
    with open(model_path, 'rb') as f_in:
        data = pickle.load(f_in)
    return data


def predict_one(booking, dv, model, features):
    """Score a single booking dict.

    Args:
        booking:  dict with feature names/values for one booking.
        dv:       fitted DictVectorizer.
        model:    trained model (XGBoost booster).
        features: list of feature names the model expects.

    Returns:
        float: probability of cancellation (in [0, 1]).
    """
    # Build a single-row dict; missing features default to 0 / 'NA'
    record = {}
    for f in features:
        record[f] = booking.get(f, 0 if isinstance(booking.get(f, 0), (int, float)) else 'NA')

    X = dv.transform([record])
    dmatrix = xgb.DMatrix(X, feature_names=dv.feature_names_)
    return float(model.predict(dmatrix)[0])


if __name__ == '__main__':
    # Quick standalone test with a sample booking
    sample = {
        'hotel': 'Resort Hotel',
        'lead_time': 50,
        'arrival_date_year': 2016,
        'arrival_date_month': 'July',
        'arrival_date_week_number': 28,
        'arrival_date_day_of_month': 1,
        'stays_in_weekend_nights': 2,
        'stays_in_week_nights': 5,
        'adults': 2,
        'children': 0,
        'babies': 0,
        'meal': 'BB',
        'country': 'PRT',
        'market_segment': 'Direct',
        'distribution_channel': 'Direct',
        'is_repeated_guest': 0,
        'previous_cancellations': 0,
        'previous_bookings_not_canceled': 0,
        'reserved_room_type': 'A',
        'assigned_room_type': 'A',
        'booking_changes': 0,
        'deposit_type': 'No Deposit',
        'agent': 0,
        'days_in_waiting_list': 0,
        'customer_type': 'Transient',
        'adr': 75.0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 0,
    }

    data = load_model('model.bin')
    proba = predict_one(sample, data['dv'], data['model'], data['features'])
    canceled = bool(proba >= 0.5)

    print(f"Probability of cancellation: {proba:.4f}")
    print(f"Predicted class: {'CANCELED' if canceled else 'NOT CANCELED'}")
