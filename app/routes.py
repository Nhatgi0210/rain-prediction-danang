from flask import Blueprint, render_template, request, jsonify
from .predictor import predict
from .weather_api import get_weather
from datetime import date, timedelta

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    today    = date.today()
    tomorrow = today + timedelta(days=1)
    return render_template('index.html',
        today    = today.strftime('%d/%m/%Y'),
        tomorrow = tomorrow.strftime('%d/%m/%Y')
    )

@bp.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    model_name = data.pop('model_name', 'Random Forest')
    result = predict(data, model_name)
    return jsonify(result)

@bp.route('/auto-fetch')
def auto_fetch():
    for_today = request.args.get('for_today', 'false').lower() == 'true'
    data, error = get_weather(for_today=for_today)
    if error:
        return jsonify({'error': error}), 500
    return jsonify(data)