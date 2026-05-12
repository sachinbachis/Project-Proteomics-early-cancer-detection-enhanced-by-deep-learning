import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Your image path
image_path = r"c:\Users\Sachin Sharma\OneDrive\Desktop\figure_heatmap_fingerprint.png.png"

# Load and display
img = mpimg.imread(image_path)
plt.figure(figsize=(12, 10))
plt.imshow(img)
plt.axis('off')
plt.title("Spectral Fingerprint Heatmap")
plt.show()