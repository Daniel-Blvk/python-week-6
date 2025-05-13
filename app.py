# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# For clean visuals
sns.set(style="whitegrid")



try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("✅ Dataset Loaded Successfully\n")

    # Display first few rows
    print("First 5 Rows:\n", df.head())

    # Data structure and info
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:\n", df.isnull().sum())

    # Clean missing values if any
    df.dropna(inplace=True)  # (not needed here, but good practice)

except FileNotFoundError:
    print(" Dataset file not found.")
except Exception as e:
    print(" Error loading dataset:", e)

# ================================
# Task 2: Basic Data Analysis
# ================================

print("\n Basic Statistics:\n", df.describe())

# Group by species and compute mean
grouped_means = df.groupby('species').mean()
print("\n Mean values by species:\n", grouped_means)

# Pattern observation
print("\n🔍 Observations:")
print("- Versicolor and Virginica have larger petal lengths/widths than Setosa.")
print("- Sepal length varies less drastically than petal length across species.")

# ================================
# Task 3: Data Visualization
# ================================

# Line Chart: Use index as pseudo-time for visualizing petal length trends
plt.figure(figsize=(10, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['petal length (cm)'], label=species)
plt.title("Petal Length Trend by Species")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)', palette='Set2')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()
