# Step 1: Import the California Housing dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch the California Housing dataset
california = fetch_california_housing()

# Step 2: Initialize the DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names)

# Step 3: Add the target variable to the DataFrame
data['PRICE'] = california.target

# Step 4: Perform Data Preprocessing (Check for missing values)
print(data.isnull().sum())  # Check if there are any missing values

# Step 5: Split dependent variable (y) and independent variables (X)
x = data.drop(['PRICE'], axis=1)
y = data['PRICE']

# Step 6: Split the data into training and testing datasets
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 7: Use Linear Regression (Train the machine)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Train the model
model = lm.fit(xtrain, ytrain)

# Step 8: Predict the y_pred for all values of train_x and test_x
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)

# Step 9: Evaluate the performance of the model for train_y and test_y
# Create a DataFrame to compare predicted and actual values for training and testing sets
train_df = pd.DataFrame({'True': ytrain, 'Predicted': ytrain_pred})
test_df = pd.DataFrame({'True': ytest, 'Predicted': ytest_pred})

# Step 10: Calculate Mean Squared Error for train_y and test_y
from sklearn.metrics import mean_squared_error, r2_score

# Calculate MSE for the train set
train_mse = mean_squared_error(ytrain, ytrain_pred)
print(f'Mean Squared Error for Train Set: {train_mse}')

# Calculate MSE for the test set
test_mse = mean_squared_error(ytest, ytest_pred)
print(f'Mean Squared Error for Test Set: {test_mse}')

# Step 11: Plotting the Linear Regression Model
plt.scatter(ytrain, ytrain_pred, c='blue', marker='o', label='Training data')
plt.scatter(ytest, ytest_pred, c='lightgreen', marker='s', label='Test data')

# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')

# Add legend
plt.legend(loc='upper left')

# Display the plot
plt.show()
