import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #used to split the data into training and testing sets
from sklearn.linear_model import LinearRegression #used to create a linear regression model
from sklearn.metrics import mean_squared_error #used to evaluate the model by comparing the predicted values to the actual values
from sklearn.preprocessing import StandardScaler #used for proper scaling of values like area, bedrooms, bathrooms, stories, parking, etc.

# Load the data
df = pd.read_csv('Housing.csv')

# Check the first few rows of the data
print(df.head())

features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
target = ['price']

data = pd.get_dummies(df, columns=features, drop_first=True) #ensures that we one-hot encode the categorical features (i.e. we convert them to binary values) and drop the first column to avoid multicollinearity (i.e. we ensure categories only have two values (yes/no) and nothing more than that)

X = data.drop(columns=target) #drops the target column and keeps all the other features
Y = np.log(data[target]) #normalizes price ranges i.e. (200000 to 100000 is log(2) = 0.693 and 400000 to 200000 is log(2) = 0.693, keeping ranges proportional) 
# so when we have data that tends to have a right skew (low cases of extremely high housing prices), we utilize logarithmic transformations to normalize the dataset (i.e. house prices, income, values that are not negative)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #test set is 20% of the data, training set is 80% of the data, random state controls shuffling data before splitting

scaler = StandardScaler() #purpose of the scaler is to ensure that all the features are on the same scale, when area and bedrooms are drastically different this ensures the model
# is able to converge more efficiently and effectively by scaling the features to a standard normal distribution (mean = 0, standard deviation = 1)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

model = LinearRegression()
model.fit(X_train_scaled, Y_train)


# Predict the values
Y_pred = model.predict(X_test_scaled)
Y_test = Y_test.values.flatten()
Y_pred = Y_pred.flatten()

#Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}') #note this is normalized on a logarithmic scale to ensure the mse "makes sense" and is not skewed by the original scale of the data

plt.scatter(Y_test, Y_pred)
min_val = min(min(Y_test), min(Y_pred))  # Get the min value to start the line
max_val = max(max(Y_test), max(Y_pred))  # Get the max value to end the line
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')  # y = x line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Manual Implementation

# X_ex = data.drop(columns=target).values
# Y_ex =  df[target].values

# a = np.zeros(X.shape[1]) # initializes each feature weight
# b = 0 # initializes the bias term/intercept
# learning_rate = 0.001 # learning rate
# epochs = 1000 # number of iterations

# for epoch in range(epochs):
#     y_pred = np.dot(X_ex, a) + b # predicted value
#     error = y_pred - Y_ex
#     cost = np.mean(error ** 2) # mean squared error

#     #Copute gradients
#     gradient_a = (2/ X.shape[0]) * np.dot(X_ex.T, error) # gradient of the weights
#     gradient_b = (2/ X.shape[0]) * np.mean(error) # gradient of the bias term

#     a -= learning_rate * gradient_a # update the weights
#     b -= learning_rate * gradient_b # update the bias term

#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}, Cost: {cost}')

# print(f'Final parameters: a = {a}, b = {b}')