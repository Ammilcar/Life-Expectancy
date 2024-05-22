# -*- coding: utf-8 -*-
"""
Amilcar Gomez Samayoa
Monday Dec 11, 2023
CDS 403 - 001

Final Project
"""

# Import Packages  # -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Function to Calculate & Print Summary Statistics  ------------------------------------
def stats(col):
    print("----- SUMMARY STATISTICS -----")
    # Mean
    mean_value = col.mean()
    print("Mean Life Expectancy:", mean_value)
    # Median
    median_value = col.median()
    print("Median Life Expectancy:", median_value)
    # Mode
    mode_value = col.mode().values
    print("Mode Life Expectancy:", mode_value)
    # Standard Deviation
    std_deviation = col.std()
    print("Standard Deviation of Life Expectancy:", std_deviation)
    # Minimum
    min_value = col.min()
    print("Minimum Life Expectancy:", min_value)
    # Maximum
    max_value = col.max()
    print("Maximum Life Expectancy:", max_value)
    print("------------------------------")

########                                 #######
########   MODIFY HERE TO LOAD IN DATA   #######
########                                 #######
### Load in Data   ------------------------------------
sumStats = pd.read_csv("/Users/Amilcar/Desktop/LEData.csv", sep=",")

### Clean the Data   ------------------------------------
# Drop rows with empty cells
sumStats_cleaned = sumStats.dropna()

# Create new CSV file with only clean data
# This file is not used in the code.  Just made for inspection outside of Python
sumStats_cleaned.to_csv('LEData_cleaned.csv', index=False)

# 627 rows removed
print("Total Rows w/ Empty Cells Removed:" ,  sumStats.shape[0] - sumStats_cleaned.shape[0], "\n")

### Present the Statistics  ------------------------------------
life_expectancy_column = sumStats_cleaned['Life expectancy']  

# Calculate & Print Summary Statistics via Function
stats(life_expectancy_column)

###  Prepare the Data # -------------------------------------------------
# Separate the Input Variable columns (x) from the Output Variable column (y)
x = sumStats_cleaned[['Alcohol', 'Hepatitis B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Schooling', 'thinness  1-19 years' ]]  # Selected features
y = sumStats_cleaned['Life expectancy']  # Output Variable


### Training the Data # --------------------------------------------------
# Split the data into training and testing sets
# 30% of data for testing, 70% for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=27)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Linear Regression Model ---------------------------------------------------------------------------------------------------------------------------------
# create a model
model = LinearRegression()
# fit the model to the training data
model.fit(x_train, y_train)
# make predictions on the test data
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

### Interpret the Model  -------------------------
# Retrieve the coefficients and intercept of the linear regression model
coefficients = model.coef_
intercept = model.intercept_
print("\n----- Lin. COEF.'S & INTERCEPT -----")
# Print the coefficients and intercept
for feature, coefficient in zip(x.columns, coefficients):
    print(f"{feature}: {coefficient}")
print("Intercept:", intercept)
print("-------------------------------")

###  ----------------------------------------------- PLOT 1
# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(8, 6))
# Plot the actual data points
ax.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
# Set labels and title
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Life Expectancy Residual Plot')
# Add a diagonal line representing perfect predictions
ax.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red', linestyle='--', label = 'Predictions')
# Add a legend
ax.legend()
# Show the plot
plt.show()

###  --------------------------------------------  PLOT 2
coefficients = model.coef_
feature_names = x.columns
indices = np.argsort(coefficients)
plt.figure(figsize=(10, 6))
plt.title('Linear Regression Coefficients')
plt.barh(range(len(indices)), coefficients[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Coefficient Value')
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Regression Decision Tree  ------------------------------------------------------------------------------------------------------------

# create model
model2 = DecisionTreeRegressor(random_state=0)
# fit model to training data
model2.fit(x_train, y_train)
# make predictions on test data
y_pred = model2.predict(x_test)
# evaluate model
mse2 = mean_squared_error(y_test, y_pred)
r22 = r2_score(y_test, y_pred)

###  ----------------------------------------------- PLOT 1
# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(8, 6))
# Plot the actual data points
ax.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
# Set labels and title
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Life Expectancy Residual Plot')
# Add a diagonal line representing perfect predictions
ax.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red', linestyle='--', label = 'Predictions')
# Add a legend
ax.legend()
# Show the plot
plt.show()

###  ----------------------------------------------- PLOT 2
importances = model2.feature_importances_
indices = np.argsort(importances)
plt.title('Regression Decision Tree Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [x.columns[i] for i in indices])
plt.xlabel('Mean Decrease in Impurity (MDI)')
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------
### Random Forest Regressor  ------------------------------------------------------------------------------------------------------------
# create model
model3 = RandomForestRegressor(random_state=0)
# fit model to training data
model3.fit(x_train, y_train)
# make predictions on test data
y_pred = model3.predict(x_test)
# evaluate model
mse3 = mean_squared_error(y_test, y_pred)
r23 = r2_score(y_test, y_pred)

###  ----------------------------------------------- PLOT 1
# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(8, 6))
# Plot the actual data points
ax.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
# Set labels and title
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Life Expectancy Residual Plot')
# Add a diagonal line representing perfect predictions
ax.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red', linestyle='--', label = 'Predictions')
# Add a legend
ax.legend()
# Show the plot
plt.show()

###  ----------------------------------------------- PLOT 2
importances = model3.feature_importances_
indices = np.argsort(importances)
plt.title('Random Forest Regressor Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [x.columns[i] for i in indices])
plt.xlabel('Mean Decrease in Impurity (MDI)')
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print the evaluation metrics
print("\n----- MEASURE PERFORMANCE -----")
print("Lin. Mean Squared Error:", mse)
print("Dec. Tree Mean Squared Error:", mse2)
print("Ran. For. Mean Squared Error:", mse3)
print("Lin. R-squared:", r2)
print("Dec. Tree R-squared:", r22)
print("Ran. For. R-squared:", r23)
print("-------------------------------")