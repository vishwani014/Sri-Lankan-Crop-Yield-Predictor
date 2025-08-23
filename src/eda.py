import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(data_path, output_dir):
    df = pd.read_csv(data_path)

    numeric_df = df.select_dtypes(include=["number"])

    # correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.savefig(f"{output_dir}/correlation.png")
    plt.close()

    # Rainfall vs. yield scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='rainfall', y='crop_yield')
    plt.savefig(f"{output_dir}/rainfall_yield.png")
    plt.close()

if __name__ == "__main__":
    run_eda("data/processed/merged_data.csv", "assets")
