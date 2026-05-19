import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def collect_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": 16.07,
        "longitude": 108.22,
        "start_date": "2015-01-01",
        "end_date": "2025-12-31",
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "relative_humidity_2m_max", "relative_humidity_2m_min",
            "wind_speed_10m_max", "pressure_msl_mean", "cloud_cover_mean"
        ]
    }

    responses = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    response = responses[0]
    daily = response.Daily()

    df = pd.DataFrame({
        "date":             pd.date_range(
                                start=pd.to_datetime(daily.Time(), unit="s"),
                                end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                                freq=pd.Timedelta(seconds=daily.Interval()),
                                inclusive="left"
                            ),
        "temp_max":         daily.Variables(0).ValuesAsNumpy(),
        "temp_min":         daily.Variables(1).ValuesAsNumpy(),
        "temp_mean":        daily.Variables(2).ValuesAsNumpy(),
        "precipitation":    daily.Variables(3).ValuesAsNumpy(),
        "humidity_max":     daily.Variables(4).ValuesAsNumpy(),
        "humidity_min":     daily.Variables(5).ValuesAsNumpy(),
        "wind_speed_max":   daily.Variables(6).ValuesAsNumpy(),
        "pressure_msl":     daily.Variables(7).ValuesAsNumpy(),
        "cloud_cover_mean": daily.Variables(8).ValuesAsNumpy(),
    })

    df.to_csv("data/danang_weather_raw.csv", index=False)
    print(f"Đã lưu dữ liệu thô: {df.shape[0]} hàng x {df.shape[1]} cột")
    return df

if __name__ == "__main__":
    collect_data()
