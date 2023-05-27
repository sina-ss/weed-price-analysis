import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Linear Regression:
#    - The linear regression line is fitted to the data points.
#    - The slope of the line indicates the estimated change in high quality price per ounce for a one-unit increase in the percentage of white people.
#    - The intercept represents the estimated high quality price per ounce when the percentage of white people is zero.

# 2. Coefficients and Intercept:
#    - The coefficient (slope) value reflects the estimated change in high quality price per ounce for a one-unit increase in the percentage of white people.
#    - The intercept represents the estimated high quality price per ounce when the percentage of white people is zero.
#    - It's important to note that in this analysis, the coefficient and intercept should be interpreted cautiously due to the lack of a strong linear relationship.

# Read the data
weed_data = pd.read_csv('weed_clean.csv')

# Select the relevant columns
data = weed_data[['percent_white', 'highQ_price_ounce']]

# Remove any rows with missing values
data = data.dropna()

# Separate the independent variable (percent_white) and dependent variable (highQ_price_ounce)
X = data['percent_white'].values.reshape(-1, 1)
y = data['highQ_price_ounce'].values

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(X, y)

# Get the coefficients and intercept of the linear regression model
slope = regression_model.coef_[0]
intercept = regression_model.intercept_

# Generate the predicted values
y_pred = regression_model.predict(X)

# Plotting the data points and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('Percentage of White People')
plt.ylabel('High Quality Price per Ounce')
plt.title('Effect of Percentage of White People on High Quality Price')
plt.legend()
plt.show()

# Print the coefficients and intercept
print('Coefficient (slope):', slope)
print('Intercept:', intercept)