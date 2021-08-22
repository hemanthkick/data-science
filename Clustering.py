"""
Clustering

"""

from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt

Iris = load_iris()

Data_iris = Iris.data 

"""
K-Mean Clustering

"""

from sklearn.cluster import KMeans

Kmns = KMeans(n_clusters = 3)
#default for n_cluster is 8, so change it
Kmns.fit(Data_iris)

Labels = Kmns.predict(Data_iris)

Ctn = Kmns.cluster_centers_

plt.scatter(Data_iris[:,2], Data_iris[:,3], c = Labels)
plt.scatter(Ctn[:,2],Ctn[:,3], marker = 'o', color = 'red', s=120)
plt.xlabel('petal length(cm)')
plt.ylabel('petal width(cm)')
plt.show()

Kmns.inertia_

K_inertia = []

for i in range(1,10):
    Kmns = KMeans(n_clusters = i, random_state = 44)
    Kmns.fit(Data_iris)
    K_inertia.append(Kmns.inertia_)

plt.plot(range(1,10),K_inertia, color = 'green', marker = 'o')
plt.xlabel('no.of K')
plt.ylabel('Inertia')
plt.show()


"""
DBScan

"""

from sklearn.cluster import DBSCAN

Dbs = DBSCAN(eps = 0.7, min_samples = 4)

Dbs.fit(Data_iris)

Labels = Dbs.labels_

plt.scatter(Data_iris[:,2], Data_iris[:,3], c = Labels)
plt.show()

"""
Hierarchical Clustering

"""
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

Hr = linkage(Data_iris, method='complete')

Dnd = dendrogram(Hr)

Labels = fcluster(Hr,4, criterion='distance')

plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.show()




