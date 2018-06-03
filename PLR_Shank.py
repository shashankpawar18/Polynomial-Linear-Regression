# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
"""As dataset is very small, we dont do feature scaling"""

# Feature Scaling
""" The Python libraries used later, do Feature Scaling by default"""

# Creating a Simple Linear Regression Model for comparison
from sklearn.linear_model import LinearRegression
SLR = LinearRegression()
SLR.fit(X,y)

# Building a PLR
from sklearn.preprocessing import PolynomialFeatures
PLR = PolynomialFeatures(degree=4)
X_poly = PLR.fit_transform(X)

make_PLR = LinearRegression()
make_PLR.fit(X_poly, y)

from sklearn.preprocessing import PolynomialFeatures
PLR2 = PolynomialFeatures(degree=5)
X_poly2 = PLR2.fit_transform(X)

make_PLR2 = LinearRegression()
make_PLR2.fit(X_poly2, y)

# Visualizing the data
sc1 = plt.scatter(X, y, color='green', marker = '.')
line1 = plt.plot(X, SLR.predict(X), color = 'blue')
line2 = plt.plot(X, make_PLR.predict(PLR.fit_transform(X)), color = 'red')
line2 = plt.plot(X, make_PLR2.predict(PLR2.fit_transform(X)), color = 'black')
plt.title('Test Results')
plt.xlabel('Position Levels')
plt.ylabel('Salary ($)')
plt.show()