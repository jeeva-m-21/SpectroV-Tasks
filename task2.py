import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Load Image ---
img = cv2.imread(r'./Resources/fruits.jpg')  # Replace with your own image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original_shape = img.shape

# --- Reshape for clustering ---
pixels_rgb = img.reshape((-1, 3)).astype(np.float32)

# --- RGB KMeans ---
k = 4
kmeans_rgb = KMeans(n_clusters=k, random_state=42)
labels_rgb = kmeans_rgb.fit_predict(pixels_rgb)
centers_rgb = np.uint8(kmeans_rgb.cluster_centers_)
segmented_rgb = centers_rgb[labels_rgb].reshape(original_shape)

# --- LAB KMeans ---
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
pixels_lab = img_lab.reshape((-1, 3)).astype(np.float32)

kmeans_lab = KMeans(n_clusters=k, random_state=42)
labels_lab = kmeans_lab.fit_predict(pixels_lab)
centers_lab = np.uint8(kmeans_lab.cluster_centers_)
segmented_lab = centers_lab[labels_lab].reshape(original_shape)
segmented_lab_rgb = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2RGB)

# --- Grayscale KMeans ---
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
pixels_gray = img_gray.reshape((-1, 1)).astype(np.float32)

kmeans_gray = KMeans(n_clusters=k, random_state=42)
labels_gray = kmeans_gray.fit_predict(pixels_gray)
centers_gray = np.uint8(kmeans_gray.cluster_centers_)
segmented_gray = centers_gray[labels_gray].reshape(img_gray.shape)

# --- Plot all ---
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_rgb)
plt.title(f"KMeans RGB Segmentation (k={k})")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(segmented_lab_rgb)
plt.title(f"KMeans LAB Segmentation (k={k})")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(segmented_gray, cmap='gray')
plt.title(f"KMeans Grayscale Segmentation (k={k})")
plt.axis('off')

plt.tight_layout()
plt.show()
