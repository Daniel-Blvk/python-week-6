
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


sns.set(style="whitegrid")



try:
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("✅ Dataset Loaded Successfully\n")

    
    print("First 5 Rows:\n", df.head())

    
    print("\nDataset Info:")
    print(df.info())

    
    print("\nMissing Values:\n", df.isnull().sum())

    
    df.dropna(inplace=True)  

except FileNotFoundError:
    print(" Dataset file not found.")
except Exception as e:
    print(" Error loading dataset:", e)


# 2: 

print("\n Basic Statistics:\n", df.describe())


grouped_means = df.groupby('species').mean()
print("\n Mean values by species:\n", grouped_means)


print("\n🔍 Observations:")
print("- Versicolor and Virginica have larger petal lengths/widths than Setosa.")
print("- Sepal length varies less drastically than petal length across species.")


# 3:

# Line Chart: 
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

# Bar Chart: 
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)', palette='Set2')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: 
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: 
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()
