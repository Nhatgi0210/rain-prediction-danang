import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

FEATURES = [
    'temp_max', 'temp_min', 'temp_mean',
    'humidity_max', 'humidity_min',
    'wind_speed_max', 'pressure_msl', 'cloud_cover_mean',
    'precipitation', 'precip_lag1', 'precip_lag2',
    'temp_range', 'month'
]

def preprocess(input_path="data/danang_weather_raw.csv",
               output_path="data/danang_weather_processed.csv"):
    df = pd.read_csv(input_path, parse_dates=['date'])

    # Tạo nhãn
    df['rain_tomorrow'] = (df['precipitation'].shift(-1) > 1.0).astype(int)
    df = df.dropna(subset=['rain_tomorrow'])
    df['rain_tomorrow'] = df['rain_tomorrow'].astype(int)

    # Feature engineering
    df['month']       = df['date'].dt.month
    df['precip_lag1'] = df['precipitation'].shift(1)
    df['precip_lag2'] = df['precipitation'].shift(2)
    df['temp_range']  = df['temp_max'] - df['temp_min']
    df = df.dropna()

    df.to_csv(output_path, index=False)
    print(f"Đã lưu dữ liệu đã xử lý: {df.shape[0]} hàng")

    X = df[FEATURES]
    y = df['rain_tomorrow']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")
    print("Đã lưu scaler vào models/scaler.pkl")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess()
