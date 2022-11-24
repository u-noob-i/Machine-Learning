import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:, : -1].values
Y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
print(classifier.predict(sc.transform([[30, 87000]])))

Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))