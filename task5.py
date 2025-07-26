import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === Step 1: Load and Preprocess Image ===
img = cv2.imread(r'./Resources/fruits.jpg')  # Replace with your image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape

# Reshape to (num_pixels, 3) and normalize
pixels = img_rgb.reshape((-1, 3)).astype(np.float32)

# === Step 2: Apply KMeans Clustering ===
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixels)
centers = np.uint8(kmeans.cluster_centers_)

# Reconstruct segmented image
segmented_pixels = centers[labels]
segmented_img = segmented_pixels.reshape((h, w, 3))

# === Step 3: Convert Clustered Output to Mask ===
# Convert segmented image to grayscale
seg_gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)

# Threshold to create binary mask
_, binary = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# === Step 4: Find Contours and Draw Boxes ===
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw overlays on original image
overlay = img_rgb.copy()
for cnt in contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    if w_box * h_box > 500:  # Ignore tiny boxes
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

# === Step 5: Show All Results ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(segmented_img)
plt.title("KMeans Segmentation (k={})".format(k))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlayed Bounding Boxes")
plt.axis('off')

plt.tight_layout()
plt.show()
