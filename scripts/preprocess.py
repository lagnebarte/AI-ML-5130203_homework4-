import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


class DataPreprocessing:

    # Get metadata
    def get_metadata(self, data):
        metadata = data.columns
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
        return metadata, numerical_cols, categorical_cols

    # Check for missing values
    def filter_missing(self, data):
        sbn.displot(
            data=data.isna().melt(value_name="missing"),
            y="variable",
            hue="missing",
            multiple="fill",
            aspect=1.5
        )
        plt.title("Missing Values by Variable")
        plt.show()

    # Plot histograms for numeric features
    def hist_frequencies(self, data, numeric_cols, bins=10):
        ncol_plots = 3
        nrow_plots = (len(numeric_cols) + ncol_plots - 1) // ncol_plots
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()

        for i, col in enumerate(numeric_cols):
            sbn.histplot(data[col], color="blue", bins=bins, ax=axs[i])
            axs[i].set_title(f"Histogram of frequencies for {col}")
            axs[i].set_xlabel(col)
            axs[i].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Correlation analysis for numerical features
    def plot_correlation(self, data, cols):
        corr = data[cols].corr()
        plt.matshow(corr, cmap="coolwarm")
        plt.xticks(range(len(cols)), cols, rotation=90)
        plt.yticks(range(len(cols)), cols)

        for (i, j), val in np.ndenumerate(corr):
            plt.text(j, i, f"{val:.1f}", ha='center', va='center', color='black')
        plt.title("Correlation Analysis")
        plt.colorbar()
        plt.show()

    # Frequency analysis for categorical features
    def get_categorical_instances(self, data, categ_cols):
        for col in categ_cols:
            print(f"\n***** {col} ******")
            print(data[col].value_counts())

    # Pie chart for a single categorical variable
    def plot_piechart(self, dataset, col):
        results = dataset[col].value_counts()
        total_samples = results.sum()
        rel_freq = results / total_samples
        sbn.set_style("whitegrid")
        plt.figure(figsize=(6, 6))
        plt.pie(rel_freq.values.tolist(), labels=rel_freq.index.tolist(), autopct='%1.1f%%')
        plt.title(f"Relative Frequency Analysis by {col}")
        plt.show()

    # Iteratively generate pie charts for multiple categorical variables
    def iter_piechart(self, dataset, categ_cols):
        ncol_plots = 2
        nrow_plots = (len(categ_cols) + ncol_plots - 1) // ncol_plots
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()

        for i, col in enumerate(categ_cols):
            results = dataset[col].value_counts()
            total_samples = results.sum()
            rel_freq = results / total_samples
            sbn.set_style("whitegrid")
            axs[i].pie(rel_freq.values.tolist(), labels=rel_freq.index.tolist(), autopct='%1.1f%%')
            axs[i].set_title(f"Relative Frequency Analysis by {col}")
        plt.tight_layout()
        plt.show()

    # Distribution of the target variable
    def plot_target_distribution(self, data, target):
        plt.figure(figsize=[8, 4])
        sbn.histplot(data[target], color='g', edgecolor="black", linewidth=2, bins=20)
        plt.title("Target Variable Distribution")
        plt.show()


"""
data_processor = DataPreprocessing()
metadata, numeric_cols, categ_cols = data_processor.get_metadata(data)
print("Metadata:", metadata)
print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categ_cols)

data_processor.filter_missing(data)
data_processor.hist_frequencies(data, numeric_cols, bins=10)
data_processor.plot_correlation(data, numeric_cols)
data_processor.iter_piechart(data, categ_cols)
"""
