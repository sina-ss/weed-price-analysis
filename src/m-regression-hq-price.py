import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# The multiple regression model is fitted using the Ordinary Least Squares (OLS) method from the `statsmodels library. The code prints the summary of the regression model, which provides information about the coefficients, standard errors, p-values, and other statistics.

# Read the data
weed_data = pd.read_csv('weed_clean.csv')

# Select the relevant columns
data = weed_data[['percent_white', 'per_capita_income', 'median_rent', 'median_age', 'highQ_price_ounce']]

# Remove any rows with missing values
data = data.dropna()

# Separate the independent variables (features) and the dependent variable (highQ_price_ounce)
X = data[['percent_white', 'per_capita_income', 'median_rent', 'median_age']]
y = data['highQ_price_ounce']

# Add a constant column to the independent variables
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Get the predicted values
y_pred = model.predict(X)

# Plotting the actual and predicted values
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual High Quality Price per Ounce')
plt.ylabel('Predicted High Quality Price per Ounce')
plt.title('Actual vs. Predicted High Quality Price per Ounce')
plt.show()
