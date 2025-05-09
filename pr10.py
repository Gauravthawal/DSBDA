# -- coding: utf-8 --
"""
Created on Tue May  6 11:26:06 2025

@author: Akash Mohalkar
"""

# Practical 10: Data Visualization using Seaborn with Iris Dataset

# Step 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the iris dataset
dataset = sns.load_dataset('iris')
print(dataset.head())  # Display first few rows

# Step 3: Plot Histograms for each feature
plt.figure(figsize=(16, 9))

plt.subplot(2, 2, 1)
sns.histplot(dataset['sepal_length'], kde=True, color='skyblue')
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
sns.histplot(dataset['sepal_width'], kde=True, color='orange')
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
sns.histplot(dataset['petal_length'], kde=True, color='green')
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
sns.histplot(dataset['petal_width'], kde=True, color='red')
plt.title('Petal Width Distribution')

plt.tight_layout()
plt.show()

# Step 4: Plot Boxplots for each feature by species
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='species', y='sepal_length', data=dataset)
plt.title('Sepal Length by Species')

plt.subplot(2, 2, 2)
sns.boxplot(x='species', y='sepal_width', data=dataset)
plt.title('Sepal Width by Species')

plt.subplot(2, 2, 3)
sns.boxplot(x='species', y='petal_length', data=dataset)
plt.title('Petal Length by Species')

plt.subplot(2, 2, 4)
sns.boxplot(x='species', y='petal_width', data=dataset)
plt.title('Petal Width by Species')

plt.tight_layout()
plt.show()
