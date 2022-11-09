import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()

df = pd.DataFrame(digits.data,columns=digits.feature_names)

df['target'] = digits.target

df['target_name'] =df.target.apply(lambda x: digits.target_names[x])

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

from sklearn.model_selection import train_test_split
 
X = df.drop(['target','target_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
model = SVC()

model.fit(X_train, y_train)
print("SVC: ", model.score(X_test, y_test))
 
model_C = SVC(C=1)
model_C.fit(X_train, y_train)
print("c = 1: ", model_C.score(X_test, y_test))
 
model_C = SVC(C=10)
model_C.fit(X_train, y_train)
print("C = 10: ", model_C.score(X_test, y_test))
 
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
print("gamma = 10: ", model_g.score(X_test, y_test))
 
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)
print("linear kernal: ", model_linear_kernal.score(X_test, y_test))

model_rbf_kernal = SVC(kernel = 'rbf', random_state = 0)
model_rbf_kernal.fit(X_train, y_train)
print("rbf kernal: ", model_rbf_kernal.score(X_test, y_test))