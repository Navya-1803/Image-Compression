import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load the image
img = cv2.imread('sample.jpg')   # Change to your image name
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Step 2: Reshape the image to a 2D array of pixels
pixels = img.reshape((-1, 3))

# Step 3: Apply K-means clustering
k = 8  # You can change this to 4, 16, etc.
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Step 4: Replace each pixel with its cluster center
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_img = compressed_pixels.reshape(img.shape).astype('uint8')

# Step 5: Display original and compressed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Compressed Image with {k} colors')
plt.imshow(compressed_img)
plt.axis('off')

plt.tight_layout()
plt.show()



# In this methord the image needs to be present in the folder where the script is run.
# We can not choose any other image from the system.
# The  image is read using OpenCV, converted to RGB format, and then processed using K-means clustering.# The number of clusters (k) can be adjusted to see how it affects the compression.
# The original and compressed images are displayed side by side using Matplotlib.
# This is a basic example and can be extended with more features like saving the compressed image