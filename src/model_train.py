import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import log

# 1. Tách train-test
def split_data(df: pd.DataFrame, target="roi", test_size=0.2, random_state=42):
    log("Tách train-test...")

    # Chọn các feature phù hợp (không dùng cột target hoặc các biến kết quả trực tiếp)
    features = ["impressions", "clicks", "conversions", "channel", "campaign",
                "cpc", "cpa", "ctr", "conversion_rate"]
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    log(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# 2. Build preprocessor
def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

# 3. Train Random Forest
def train_random_forest(X_train, y_train):
    log("Train RandomForestRegressor with preprocessor...")

    preprocessor = build_preprocessor(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    log("Random Forest training completed.")

    return pipeline

# 4. Train Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    log("Train GradientBoostingRegressor with preprocessor...")

    preprocessor = build_preprocessor(X_train)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    log("Gradient Boosting training completed.")

    return pipeline

# 5. Gói train 2 model + lưu file
def train_models(df_clean: pd.DataFrame, output_dir="outputs/models/"):
    log("Bắt đầu train 2 mô hình…")

    # Tách dữ liệu
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # Train 2 mô hình
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu mô hình
    joblib.dump(rf_model, f"{output_dir}/random_forest.pkl")
    joblib.dump(gb_model, f"{output_dir}/gradient_boosting.pkl")

    log(f"Đã lưu mô hình vào thư mục {output_dir}")

    return rf_model, gb_model, X_test, y_test
