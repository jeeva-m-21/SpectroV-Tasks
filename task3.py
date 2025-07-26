from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load dataset
digits = load_digits()
X = digits.data         # Shape: (1797, 64)
y = digits.target       # Labels (0–9)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X)


import seaborn as sns

plt.figure(figsize=(12, 5))

# PCA
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', legend='full', s=40)
plt.title("PCA Projection (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')

# t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', legend='full', s=40)
plt.title("t-SNE Projection (2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
