import pandas as pd 
df= pd.read_csv('student data.csv')
df

df.pop('Name')
df

df.pop('Class')
df

df.min()
df.max()
df.median()
df.std()
df.mean()

df['Marks'].mean()
df['Marks'].std()
df['Marks'].max()


grp1 = df.groupby('Marks')
top = grp1.get_group(90)
top
grp1.groups

import seaborn as sns
df = sns.load_dataset('iris') 
df
df.describe()

gr = df.groupby('species')
se = gr.get_group('setosa')
se

se.shape
se.describe()
