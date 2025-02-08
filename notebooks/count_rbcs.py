import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
# image_path = 'data/raw/AneRBC-I/Healthy_individuals/Original_images/001_h.png'
# mask_path = 'data/raw/AneRBC-I/Healthy_individuals/Binary_segmented/001_h.png'
image_path = 'data/raw/AneRBC-I/Anemic_individuals/Original_images/002_a.png'
mask_path = 'data/raw/AneRBC-I/Anemic_individuals/Binary_segmented/002_a.png'
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

binary_mask = np.where(mask == 255, 0, 1).astype(np.uint8)

# Detect contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Estimated number of RBCs: {len(contours)}")

# Draw the contours on the image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Draw contours in green

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected RBCs")
plt.axis("off")
plt.show()
