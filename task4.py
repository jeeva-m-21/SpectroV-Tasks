import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load sensor data (561 features per sample)
df = pd.read_csv(r'./Resources/X_train.txt', delim_whitespace=True, header=None)
print("Shape:", df.shape)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

from sklearn.cluster import KMeans, DBSCAN

# KMeans
kmeans = KMeans(n_clusters=6, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot KMeans
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_kmeans, palette='Set1', s=15)
plt.title("KMeans Clustering (PCA)")

# Plot DBSCAN
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_dbscan, palette='Set2', s=15)
plt.title("DBSCAN Clustering (PCA)")

plt.tight_layout()
plt.show()
