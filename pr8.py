# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:45:06 2025

@author: gaura
"""
# Practical 8: Seaborn visualizations on Titanic dataset

import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
dataset = sns.load_dataset('titanic')

# Display first few rows
print(dataset.head())

# a. Histogram / Distribution plot of Age with KDE
sns.histplot(data=dataset, x='age', bins=10, kde=True)
plt.title('Distribution Plot of Age with KDE')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# a. Histogram / Distribution plot of Age without KDE
sns.histplot(data=dataset, x='age', bins=10, kde=False)
plt.title('Distribution Plot of Age without KDE')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# b. Countplot of Sex
sns.countplot(data=dataset, x='sex')
plt.title('Count of Passengers by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# c. Boxplot of Age vs Pclass
sns.boxplot(data=dataset, x='pclass', y='age')
plt.title('Boxplot of Age by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# d. Violinplot of Age vs Pclass
sns.violinplot(data=dataset, x='pclass', y='age')
plt.title('Violinplot of Age by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# e. Stripplot of Age vs Pclass
sns.stripplot(data=dataset, x='pclass', y='age', jitter=True)
plt.title('Stripplot of Age by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# f. Swarmplot of Age vs Sex
sns.swarmplot(data=dataset, x='sex', y='age')
plt.title('Swarmplot of Age by Sex')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()

# g. Heatmap of Correlation Matrix
corr_matrix = dataset.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
