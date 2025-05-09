import pandas as pd 
df= pd.read_csv('data.csv')
df
df.dtypes
df.describe()
df.groupby('Age').size()
df.isna()
df.isna().sum()
df['Age'] = df['Age'].astype(int)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
print(df['Age'].dtype)
print(df['BMI'].dtype)
df.sort_values('Age', ascending = False)
df.rename(columns = {'Age' : 'year old'})
# df.drop(column =['Age'])
df.drop_duplicates()
df.head()
df.tail()
df.count()
df['Age'] = pd.to_numeric(df['Age'])
df['Age']
# to convert categorical data into numerical in pandas
pd.get_dummies(df['Classification'])