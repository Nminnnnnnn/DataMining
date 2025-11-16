import os
import joblib
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def log(msg: str):
    """In ra log có timestamp."""
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(path):
        os.makedirs(path)
        log(f"Tạo thư mục: {path}")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset CSV.
    Nếu có lỗi — log và raise.
    """
    try:
        df = pd.read_csv(path)
        log(f"Dataset loaded: {path} — shape {df.shape}")
        return df
    except Exception as e:
        log(f"Lỗi load dataset: {e}")
        raise e


def save_fig(fig, filename: str, dpi=300):
    """
    Lưu figure matplotlib.
    """
    ensure_dir(os.path.dirname(filename))
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"Đã lưu biểu đồ: {filename}")


def save_model(model, path: str):
    """
    Lưu model (.pkl).
    """
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)
    log(f"Đã lưu model: {path}")
