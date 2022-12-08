import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customer ID')
plt.ylabel('Eucledian Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_h = hc.fit_predict(X)

plt.scatter(X[y_h == 0, 0], X[y_h == 0, 1], s = 100, c = 'red', label = 'CLuster 1')
plt.scatter(X[y_h == 1, 0], X[y_h == 1, 1], s = 100, c = 'blue', label = 'CLuster 2')
plt.scatter(X[y_h == 2, 0], X[y_h == 2, 1], s = 100, c = 'green', label = 'CLuster 3')
plt.scatter(X[y_h == 3, 0], X[y_h == 3, 1], s = 100, c = 'cyan', label = 'CLuster 4')
plt.scatter(X[y_h == 4, 0], X[y_h == 4, 1], s = 100, c = 'magenta', label = 'CLuster 5')
plt.scatter(hc.cluster_centers_[:, 0], hc.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()