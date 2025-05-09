import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns  # Add this line
df = pd.read_csv('student data.csv')
df.shape
df.describe()
df.head()
df.info()
df.isnull()
df['Class'].fillna(df['Class'].mode()) 
df.isnull().sum()
df.fillna(0)
df.fillna(method ='backfill')
df.fillna(method ='pad')

df.dropna(axis = 0 , inplace = True)
df
df.dropna(axis = 0, how ='all')
df

# We use method = 'bfill’ for taking values from the next row
df['Age'].fillna(method='bfill', inplace=True) 
print(df['Age'])

#method = 'pad’ for taking values from the previous row 
df['Class'].fillna(method='pad', inplace=True)
print(df['Class'].head(10))

# Only do boxplot for numeric column 'Age'
plt.boxplot(df['Age'].dropna())  # drop NaN just in case
plt.title('Boxplot of Age')
plt.show()

# For 'Class' (categorical), use countplot
sns.countplot(x='Class', data=df)
plt.title('Count of Students in Each Class')
plt.show()

# -----------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X= np.array([2,4,6,8,10,12])
np.mean(X)
np.median(X)

X= np.array([2,4,6,8,10,12])
df= pd.DataFrame(X)
print (df)

plt.boxplot(X)
df.plot.box()


data = {
    "Name": ["Amit", "Priya", "Raj", "Sneha", "Vikram", "Ananya", "Rohan"],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Marks": [85, 80, 78, np.nan, 76, 82, np.nan],
    "Age": [np.nan,21,22,np.nan,24,np.nan,26]
}
df = pd.DataFrame(data)
print(df)

df.head()
df.tail()
df.count()
df.isnull()

df.isnull().sum()
df.dropna()
df.fillna(0)

df['Marks'].fillna(df['Marks'].mean())
df['Age'].fillna(df['Age'].median())

df.fillna(method='bfill')
df.fillna(method='pad')





Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Age'] < lower) | (df['Age'] > upper)]
df = df[(df['Age'] >= lower) & (df['Age'] <= upper)]
df


x = df[['Age' , 'Marks']]
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(x)
pd.DataFrame(x_scaled).describe() 



from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x) 
pd.DataFrame(x_scaled).describe() 
