# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:52:25 2025

@author: gaura
"""
import matplotlib.pyplot as plt
import seaborn as sns 
df = sns.load_dataset('titanic')
df
df = df[['sex','age','survived']]
df
sns.boxplot(x='sex',y='age',data=df) 
plt.show()
sns.boxplot(x='sex',y='age',hue='survived',data = df)
plt.show()
