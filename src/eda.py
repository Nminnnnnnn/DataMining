import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils import log

# 1. Thống kê mô tả
def describe_data(df: pd.DataFrame):
    log("Thống kê mô tả dữ liệu:")
    print(df.describe(include="all").T)
    print("\n\n")
    df.describe(include="all").T.to_csv("outputs/numeric_summary.csv")

# 2. Heatmap tương quan numeric
def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    log("Vẽ heatmap tương quan numeric...")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# 3. Boxplot ROI theo Category 
def plot_boxplot(df: pd.DataFrame, cat_col, save_path=None):
    log(f"Vẽ Boxplot ROI theo {cat_col}...")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=cat_col, y="roi")
    plt.title(f"ROI Distribution by {cat_col}")
    plt.xticks(rotation=45)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# 4. Scatter ROI vs Numeric Features
def plot_scatter_roi(df: pd.DataFrame, x_col, save_path=None):
    log(f"Vẽ Scatter ROI vs {x_col}...")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y="roi")
    plt.title(f"ROI vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel("ROI")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# 5. Distribution ROI
def plot_roi_distribution(df: pd.DataFrame, save_path=None):
    log("Vẽ phân phối ROI...")

    plt.figure(figsize=(8, 6))
    sns.histplot(df["roi"], kde=True, bins=30)
    plt.title("Distribution of ROI")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# 6. Chạy toàn bộ EDA
def run_full_eda(df: pd.DataFrame, output_dir="outputs/eda/"):
    log("Bắt đầu EDA...")

    os.makedirs(output_dir, exist_ok=True)

    describe_data(df)
    plot_correlation_heatmap(df, os.path.join(output_dir, "corr_heatmap.png"))
    for cat_col in ["channel", "campaign"]:
        plot_boxplot(df, cat_col, os.path.join(output_dir, f"roi_by_{cat_col}.png"))

    # Scatter ROI vs numeric features quan trọng
    for num_col in ["cpa", "cpc", "cost"]:
        plot_scatter_roi(df, num_col, os.path.join(output_dir, f"roi_vs_{num_col}.png"))
    plot_roi_distribution(df, os.path.join(output_dir, "roi_distribution.png"))

    log("🎉 Hoàn thành EDA!")
