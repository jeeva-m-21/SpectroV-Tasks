import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

# Generate datasets
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_blobs)

plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_kmeans, cmap='viridis', s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='red', marker='X', s=100, label='Centroids')
plt.title("KMeans Clustering (Blobs)")
plt.legend()
plt.show()

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_dbscan, cmap='plasma', s=30)
plt.title("DBSCAN Clustering (Moons)")
plt.show()
