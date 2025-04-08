# Import necessary libraries for data analysis, visualization, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset from CSV file
customers = pd.read_csv('Ecommerce Customers.csv')

# Display initial data exploration
print(customers.head())          # Print first 5 rows of the dataset
print(customers.describe())      # Show summary statistics of numerical columns
print(customers.info())          # Display data types and check for missing values

# Set Seaborn style for enhanced visualization aesthetics
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# Create visualizations to explore relationships
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)  # Scatter plot: Website time vs Spending
plt.show()

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)      # Scatter plot: App time vs Spending
plt.show()

sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)  # Hexbin plot: App time vs Membership
plt.show()

sns.pairplot(customers)  # Pairplot to visualize all numerical variable relationships
plt.show()

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)  # Linear fit plot: Membership vs Spending
plt.show()

# Define features (X) and target variable (y)
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]  # Input features
y = customers['Yearly Amount Spent']  # Target variable to predict

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Initialize and train the Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)  # Fit the model using training data

# Print the model coefficients
print('Coefficients: \n', lm.coef_)

# Generate predictions on the test set
predictions = lm.predict(X_test)

# Plot a scatter of actual vs predicted values
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (Actual Values)')
plt.ylabel('Predicted Y')
plt.title('Actual vs Predicted Spending')
plt.show()

# Calculate and print evaluation metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))   # Mean Absolute Error
print('MSE:', metrics.mean_squared_error(y_test, predictions))    # Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # Root Mean Squared Error

# Visualize the distribution of prediction errors (residuals)
sns.histplot((y_test - predictions), bins=50, kde=True)  # Histogram with KDE curve for residuals
plt.xlabel('Prediction Errors')
plt.title('Distribution of Residuals')
plt.show()

# Create a DataFrame to display feature coefficients
coefficients = pd.DataFrame(lm.coef_, X.columns)
coefficients.columns = ['Coefficient']
print(coefficients)  # Display coefficients to interpret feature importance