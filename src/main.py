import os
import pandas as pd

# Import internal modules
from src.preprocessing import load_and_clean, split_data
from src.eda import run_full_eda
from src.model_train import train_models
from src.evaluate import evaluate_models
from src.utils import log, ensure_dir

#   MAIN PIPELINE
def main():

    log("BẮT ĐẦU CHẠY PIPELINE DỰ ÁN MARKETING ROI PREDICTION")

    # 1. Load dataset
    data_path = "data/media_all_channels.csv"

    log(f"Đang load dataset từ: {data_path}")
    df = pd.read_csv(data_path)
    df = load_and_clean(df)  

    log(f"Dataset có {df.shape[0]} dòng và {df.shape[1]} cột")

    # 2. Thực hiện EDA cơ bản + biểu đồ
    log("Chạy EDA (describe + heatmap + boxplot + scatter + dist)…")
    ensure_dir("outputs/plots")
    run_full_eda(df, output_dir="outputs/plots")

    # 3. Train Random Forest & Gradient Boosting
    log("Train mô hình…")
    rf_model, gb_model, X_test, y_test = train_models(df, output_dir="outputs/models/")

    # 4. Evaluate
    log("Đánh giá mô hình…")
    summary = evaluate_models(rf_model, gb_model, X_test, y_test)

    # 5. Xuất bảng so sánh model
    ensure_dir("outputs")
    summary_path = "outputs/model_comparison.csv"
    summary.to_csv(summary_path, index=False)

    log(f"Đã lưu bảng so sánh model tại: {summary_path}")

    log("PIPELINE HOÀN TẤT — TẤT CẢ ĐÃ CHẠY THÀNH CÔNG!")

#   RUN SCRIPT
if __name__ == "__main__":
    main()
