import os
import numpy as np
import tifffile
from sklearn.mixture import GaussianMixture
import cv2

def select_file(title="Select file"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title)
    return file_path

# Change the working directory
desired_path = 'path/to/your/directory'  # Replace with your desired path
os.chdir(desired_path)

# Select and read the background file
background_file = select_file("Select the background file")
background_image = tifffile.imread(background_file)

# Select and read the stack file
stack_file = select_file("Select the stack file")
with tifffile.TiffFile(stack_file) as tif:
    stack = tif.asarray()

    # Subtract background from each slice
    processed_stack = stack.astype(np.int32) - background_image.astype(np.int32)
    processed_stack[processed_stack < 0] = 0  # Set negative values to 0
    processed_stack = processed_stack.astype(np.uint16)

# Preprocess the data
num_frames, height, width = processed_stack.shape
pixels = processed_stack.reshape(num_frames, -1).T

# Create and fit the GMM
n_clusters = 5
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(pixels)

# Cluster the pixels
cluster_labels = gmm.predict(pixels)

# Set up the SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50  # Adjust this based on your data
detector = cv2.SimpleBlobDetector_create(params)

# Find the cluster with the largest/most prominent blob
most_prominent_blob_cluster = None
largest_blob_size = 0
roi = None

for i in range(n_clusters):
    # Create an image for each cluster
    cluster_image = np.zeros(height * width, dtype=np.uint8)
    cluster_image[cluster_labels == i] = 255
    cluster_image = cluster_image.reshape(height, width)

    # Detect blobs
    keypoints = detector.detect(cluster_image)

    # Find the largest blob in this cluster
    if keypoints:
        largest_blob = max(keypoints, key=lambda k: k.size)
        if largest_blob.size > largest_blob_size:
            largest_blob_size = largest_blob.size
            most_prominent_blob_cluster = i
            # Define the ROI based on the blob's position and size
            x, y = largest_blob.pt
            r = largest_blob.size / 2
            roi = (int(x - r), int(y - r), int(2 * r), int(2 * r))

# Check if a cluster with the most prominent blob was found
if most_prominent_blob_cluster is None or roi is None:
    print("No cluster with a prominent blob found.")
    raise ValueError("No cluster with a prominent blob found")

# Measure Pixel Values in the ROI Across All Frames
x, y, w, h = roi
pixel_values_over_time = []
for frame in processed_stack:
    roi_frame = frame[y:y+h, x:x+w]
    pixel_values_over_time.append(np.sum(roi_frame))

# [Continue with processing the pixel values, e.g., plotting, analysis, etc.]
