import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1: -1]
Y = dataset.iloc[:, -1]

Y = Y.values.reshape(len(Y), 1)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
X = scX.fit_transform(X)
Y = scY.fit_transform(Y)

from sklearn.svm import SVR
regressor = SVR(kernel=('rbf'))
regressor.fit(X, Y)

print(scY.inverse_transform(regressor.predict(scX.transform([[6.5]])).reshape(-1, 1)))

plt.scatter(scX.inverse_transform(X), scY.inverse_transform(Y), color = 'red')
plt.plot(scX.inverse_transform(X), scY.inverse_transform(regressor.predict(X).reshape(-1, 1)), color = 'blue')
plt.title('SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()