import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils import log

# 1. HÀM LOAD + CLEAN DATA
def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dataset cơ bản:
    - Xóa dấu $ và dấu , trong Acquisition_Cost
    - Chuyển ROI sang số
    - Chuyển Date thành datetime
    - Xử lý các cột Duration
    """
    df = df.copy()

    # Clean Acquisition_Cost
    if "Acquisition_Cost" in df.columns:
        df["Acquisition_Cost"] = (
            df["Acquisition_Cost"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"], errors="coerce")

    # Clean Duration
    if "Duration" in df.columns:
        df["Duration"] = (
            df["Duration"].astype(str).str.replace("days", "", regex=False).str.strip()
        )
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

    # Clean ROI
    if "ROI" in df.columns:
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")

    # Convert Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    log("Dataset đã được làm sạch")
    return df


# 2. TẠO PREPROCESSING PIPELINE
def build_preprocessor(df: pd.DataFrame, target_col: str):
    """
    Xây dựng transformer cho numeric & categorical.
    """
    # Tách X và y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Phân loại cột
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    log(f"Numeric columns: {numerical_cols}")
    log(f"Categorical columns: {categorical_cols}")

    # Pipeline cho numeric
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline cho categorical
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Kết hợp
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return X, y, preprocessor


# 3. SPLIT DATA
def split_data(X, y, test_size=0.2, random_state=42):
    log("Đang chia train/test...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
