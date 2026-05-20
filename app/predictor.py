import joblib
import numpy as np
import os

# Load models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

models = {
    'Logistic Regression': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl')),
    'Decision Tree':       joblib.load(os.path.join(MODELS_DIR, 'decision_tree.pkl')),
    'Random Forest':       joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl')),
    'Voting Classifier':   joblib.load(os.path.join(MODELS_DIR, 'voting_classifier.pkl')),
}
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

FEATURES = [
    'temp_max', 'temp_min', 'temp_mean',
    'humidity_max', 'humidity_min',
    'wind_speed_max', 'pressure_msl', 'cloud_cover_mean',
    'precipitation', 'precip_lag1', 'precip_lag2',
    'temp_range', 'month'
]

def predict(data: dict, model_name: str = 'Random Forest'):
    # Tính các feature tự động
    data['temp_range']  = data['temp_max'] - data['temp_min']
    data['precip_lag1'] = data.get('precip_lag1', 0)
    data['precip_lag2'] = data.get('precip_lag2', 0)

    X = np.array([[data[f] for f in FEATURES]])

    model = models[model_name]

    # Logistic Regression cần chuẩn hóa
    if model_name == 'Logistic Regression':
        X = scaler.transform(X)

    prob    = model.predict_proba(X)[0][1]
    result  = int(prob >= 0.5)

    return {
        'result':      result,
        'probability': round(float(prob) * 100, 1),
        'label':       '🌧️ Có mưa' if result == 1 else '☀️ Không mưa',
        'model':       model_name
    }