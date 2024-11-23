import os
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class ChartFactory:
    @staticmethod
    def create_chart(chart_type, **kwargs):
        if chart_type == "bar":
            return BarChart(**kwargs)
        elif chart_type == "heatmap":
            return Heatmap(**kwargs)
        elif chart_type == "pairplot":
            return PairPlot(**kwargs)
        elif chart_type == "boxplot":
            return Boxplot(**kwargs)
        elif chart_type == "pie":
            return PieChart(**kwargs)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")



class BarChart:
    def __init__(self, data, column, title="Bar Chart", output_path="visualizations"):
        self.data = data
        self.column = column
        self.title = title
        self.output_path = output_path

    def plot(self):
        counts = self.data[self.column].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(self.title)
        plt.xlabel(self.column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/{self.column}_bar_chart.png")
        plt.close()


class Heatmap:
    def __init__(self, data, title="Heatmap", output_path="visualizations"):
        self.data = data
        self.title = title
        self.output_path = output_path

    def plot(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(self.title)
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/heatmap.png")
        plt.close()


class PairPlot:
    def __init__(self, data, title="Pairplot", output_path="visualizations"):
        self.data = data
        self.title = title
        self.output_path = output_path

    def plot(self):
        sns.pairplot(self.data)
        plt.title(self.title)
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/pairplot.png")
        plt.close()


class Boxplot:
    def __init__(self, data, column, title="Boxplot", output_path="visualizations"):
        self.data = data
        self.column = column
        self.title = title
        self.output_path = output_path

    def plot(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data[self.column])
        plt.title(self.title)
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/{self.column}_boxplot.png")
        plt.close()


class PieChart:
    def __init__(self, data, column, title="Pie Chart", output_path="visualizations"):
        self.data = data
        self.column = column
        self.title = title
        self.output_path = output_path

    def plot(self):
        counts = self.data[self.column].value_counts()
        plt.figure(figsize=(8, 8))
        counts.plot.pie(autopct="%1.1f%%")
        plt.title(self.title)
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/{self.column}_pie_chart.png")
        plt.close()


class DataVisualizerFacade:
    def __init__(self, output_path="visualizations"):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def visualize_data_evaluation(self, data, top_n=10):

        print("[Visualization] Evaluating data structure...")

        column = data.columns[0]
        counts = data[column].value_counts().head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("Initial Data Distribution (Top Categories)")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        bar_chart_path = f"{self.output_path}/initial_data_distribution_bar_chart.png"
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(bar_chart_path)
        plt.close()
        print(f"[Visualization] Bar chart saved to {bar_chart_path}")


        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_path = f"{self.output_path}/correlation_heatmap.png"
            plt.savefig(heatmap_path)
            plt.close()
            print(f"[Visualization] Heatmap saved to {heatmap_path}")
        else:
            print("[Visualization] Skipping heatmap. Not enough numeric columns.")

    def visualize_clean(self, before_data, after_data, column, top_n=10):

        print("[Visualization] Visualizing clean step...")


        before_counts = before_data[column].value_counts().head(top_n)
        after_counts = after_data[column].value_counts().head(top_n)

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        before_counts.plot(kind="bar", color="blue", title="Before Cleaning")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        after_counts.plot(kind="bar", color="green", title="After Cleaning")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/clean_step_comparison.png", bbox_inches="tight")
        plt.close()

    def visualize_data_transformation(self, before_data, after_data):
        print("[Visualization] Visualizing data transformation...")

        before_plot = ChartFactory.create_chart(
            "pairplot", data=before_data, title="Before Transformation", output_path=self.output_path
        )
        before_plot.plot()
        after_plot = ChartFactory.create_chart(
            "pairplot", data=after_data, title="After Transformation", output_path=self.output_path
        )
        after_plot.plot()

    def visualize_class_balance(self, data, column):
        print("[Visualization] Visualizing class balance...")
        pie_chart = ChartFactory.create_chart(
            "pie", data=data, column=column, title="Class Balance", output_path=self.output_path
        )
        pie_chart.plot()

    def visualize_outliers(self, data, column):

        print(f"[Visualization] Visualizing outliers for column '{column}'...")

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")
        try:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        except Exception as e:
            raise ValueError(f"Failed to convert column '{column}' to numeric: {str(e)}")

        data = data.dropna(subset=[column])

        if data[column].empty:
            raise ValueError(f"Column '{column}' has no valid numeric data after conversion.")

        plt.figure(figsize=(10, 6))
        sns.boxplot(data[column])
        plt.title(f"Outliers Detection for '{column}'")
        plt.xlabel(column)
        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(f"{self.output_path}/{column}_outliers_detection.png")
        plt.close()

    def visualize_feature_relationships(self, data, feature_definitions):
        print("[Visualization] Visualizing feature relationships...")

        for new_feature, base_features in feature_definitions.items():
            if new_feature in data.columns and all(f in data.columns for f in base_features):
                plt.figure(figsize=(10, 6))
                for base_feature in base_features:
                    plt.scatter(data[base_feature], data[new_feature], label=base_feature, alpha=0.7)
                plt.title(f"Relationship of {new_feature} with base features")
                plt.xlabel("Base Features")
                plt.ylabel(new_feature)
                plt.legend()
                file_path = f"{self.output_path}/{new_feature}_relationships.png"
                plt.savefig(file_path)
                plt.close()
                print(f"[Visualization] Relationship plot saved to {file_path}")
            else:
                print(f"[Visualization] Feature '{new_feature}' or its base features not found in data.")
