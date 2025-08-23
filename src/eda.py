import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(data_path):
    df = pd.read_csv(data_path)

    df['season'] = pd.Categorical(df['season'], categories=['Maha', 'Yala'], ordered=True)

    # Plot 1: Yield over time by season
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x='Year', y='Avg_Yield_Kg_Ha', hue='season', marker='o')
    plt.title('Rice Yield (Kg/Ha) Over Time by Season')
    plt.xlabel('Year')
    plt.ylabel('Average Yield (Kg/Ha)')
    plt.savefig('assets/yield_over_time.png')
    plt.close()

    # Plot 2: Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[['Avg_Yield_Kg_Ha', 'Sown_Ha', 'Sown_to_Harvest_Ratio', 'avg_price_lkr', 'rfh_avg', 'Population', 'Inflation']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Features')
    plt.savefig('assets/correlation_heatmap.png')
    plt.close()

    # Plot 3: Yield vs. Rainfall (rfh_avg)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='rfh_avg', y='Avg_Yield_Kg_Ha', hue='season', size='Inflation')
    plt.title('Rice Yield vs. Average Rainfall by Season')
    plt.xlabel('Average Rainfall (mm)')
    plt.ylabel('Average Yield (Kg/Ha)')
    plt.savefig('assets/yield_vs_rainfall.png')
    plt.close()

    # Plot 4: Boxplot of yield by season
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='season', y='Avg_Yield_Kg_Ha')
    plt.title('Rice Yield Distribution by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Yield (Kg/Ha)')
    plt.savefig('assets/yield_boxplot.png')
    plt.close()

    # Plot 5: Prices over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Year', y='avg_price_lkr', hue='season')
    plt.title('Average Rice Price (LKR) Over Time')
    plt.xlabel('Year')
    plt.ylabel('Price (LKR)')
    plt.savefig('assets/prices_over_time.png')
    plt.close()

    # Plot 6: Yield distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Avg_Yield_Kg_Ha', hue='season', fill=True, common_norm=False, alpha=0.5)
    plt.title('Distribution of Average Yield (Kg/Ha)')
    plt.xlabel('Average Yield (Kg/Ha)')
    plt.ylabel('Frequency')
    plt.savefig('assets/yield_distribution.png')
    plt.close()


if __name__ == "__main__":
    run_eda("data/processed/merged_data.csv")

    print("EDA plots saved to 'assets' directory")
