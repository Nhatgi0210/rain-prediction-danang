import requests
from datetime import date, timedelta

def get_weather(for_today=False):
    """
    for_today=True  → lấy dữ liệu hôm qua để dự đoán hôm nay
    for_today=False → lấy dữ liệu hôm nay để dự đoán ngày mai
    """
    today     = date.today()
    yesterday = today - timedelta(days=1)
    two_days  = today - timedelta(days=2)
    three_days = today - timedelta(days=3)

    # Nếu dự đoán hôm nay → input là hôm qua
    target      = yesterday if for_today else today
    lag1_date   = two_days  if for_today else yesterday
    lag2_date   = three_days if for_today else two_days

    def fetch_day(d):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   16.07,
            "longitude":  108.22,
            "start_date": str(d),
            "end_date":   str(d),
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "relative_humidity_2m_max",
                "relative_humidity_2m_min", "wind_speed_10m_max",
                "pressure_msl_mean", "cloud_cover_mean"
            ]
        }
        r = requests.get(url, params=params)
        d_data = r.json()['daily']
        return {
            'temp_max':         d_data['temperature_2m_max'][0],
            'temp_min':         d_data['temperature_2m_min'][0],
            'temp_mean':        d_data['temperature_2m_mean'][0],
            'precipitation':    d_data['precipitation_sum'][0],
            'humidity_max':     d_data['relative_humidity_2m_max'][0],
            'humidity_min':     d_data['relative_humidity_2m_min'][0],
            'wind_speed_max':   d_data['wind_speed_10m_max'][0],
            'pressure_msl':     d_data['pressure_msl_mean'][0],
            'cloud_cover_mean': d_data['cloud_cover_mean'][0],
        }

    try:
        main_data  = fetch_day(target)
        lag1_data  = fetch_day(lag1_date)
        lag2_data  = fetch_day(lag2_date)

        main_data['precip_lag1'] = lag1_data['precipitation']
        main_data['precip_lag2'] = lag2_data['precipitation']
        main_data['month']       = target.month

        return main_data, None
    except Exception as e:
        return None, str(e)