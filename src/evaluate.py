import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import log, ensure_dir, save_fig

# 1. Evaluate 1 model
def evaluate_model(model, X_test, y_test):
    log("📊 Đang đánh giá mô hình...")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    log(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "preds": preds
    }

# 2. Vẽ: Actual vs Predicted
def plot_actual_vs_pred(y_test, preds, filename="outputs/plots/actual_vs_pred.png"):

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, preds, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", linewidth=2)
    plt.xlabel("Actual ROI")
    plt.ylabel("Predicted ROI")
    plt.title("Actual vs Predicted")

    save_fig(fig, filename)

# 3. Residual Plot
def plot_residuals(y_test, preds, filename="outputs/plots/residuals.png"):

    residuals = y_test - preds

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(preds, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted ROI")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    save_fig(fig, filename)

# 4. Feature Importance (cho tree model)
def plot_feature_importance(model, feature_names, filename="outputs/plots/feature_importance.png"):

    if not hasattr(model, "feature_importances_"):
        log("Model này không hỗ trợ Feature Importance.")
        return

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig = plt.figure(figsize=(8, 6))
    plt.barh(range(len(importance)), importance[sorted_idx])
    plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance")

    save_fig(fig, filename)

# 5. Evaluate 2 models
def evaluate_models(rf_model, gb_model, X_test, y_test):

    log("Bắt đầu đánh giá Random Forest")
    rf_result = evaluate_model(rf_model, X_test, y_test)

    log("Bắt đầu đánh giá Gradient Boosting")
    gb_result = evaluate_model(gb_model, X_test, y_test)

    # Create summary table
    summary_df = pd.DataFrame({
        "Model": ["RandomForest", "GradientBoosting"],
        "MAE": [rf_result["MAE"], gb_result["MAE"]],
        "RMSE": [rf_result["RMSE"], gb_result["RMSE"]],
        "R2": [rf_result["R2"], gb_result["R2"]],
    })

    log("Bảng so sánh model:\n" + str(summary_df))

    ensure_dir("outputs/plots")

    # Save plots for RF model only (you can extend for GB too)
    plot_actual_vs_pred(y_test, rf_result["preds"], "outputs/plots/actual_vs_pred_rf.png")
    plot_residuals(y_test, rf_result["preds"], "outputs/plots/residuals_rf.png")
    plot_feature_importance(rf_model, X_test.columns, "outputs/plots/feature_importance_rf.png")

    return summary_df
