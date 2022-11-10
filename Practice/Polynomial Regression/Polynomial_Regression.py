import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linreg2 = LinearRegression()
linreg2.fit(X_poly, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X, linreg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, Y, color = 'red')
plt.plot(X, linreg2.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print("Linear Regression: ", linreg.predict([[6.5]]))
print("Polynomial Regressoin: ", linreg2.predict(poly_reg.fit_transform([[6.5]])))