#!/usr/bin/env python
# coding: utf-8

# In[62]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[63]:


iou_final = []


# In[64]:


# Read the image
image = cv2.imread('C:\\Users\\FARAH\\Desktop\\Digtial image processing\\Dataset_Pre\\output_pre\\baso\\1.bmp', cv2.IMREAD_COLOR)
plt.imshow(image)


# In[65]:


# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the purple color in the HSV color space
lower_bound = np.array([129, 40, 0])
upper_bound = np.array([174, 255, 255])

# Create the mask
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Extract the purple pixels using the mask
purple = cv2.bitwise_and(image, image, mask=mask)

# print image
plt.imshow(purple,cmap="gray")


# In[66]:


# Remove noise
purple = cv2.medianBlur(purple, 5)
plt.imshow(purple,cmap="gray")


# In[67]:


# Convert to grayscale and apply thresholding
gray = cv2.cvtColor(purple, cv2.COLOR_BGR2GRAY)
# Threshold the image to create a topographic map
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)
plt.imshow(thresh,cmap="gray")


# In[68]:


# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 6))
# Opening, Closing
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
plt.imshow(closed,cmap="gray")


# In[70]:


# Erosion
eroded = cv2.erode(closed, kernel, iterations=2)
# Dilation
dilated = cv2.dilate(eroded, kernel, iterations=2)
plt.imshow(dilated,cmap="gray")


# In[71]:


# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_area = 100  # Minimum area threshold for white blood cells
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Create a mask for white blood cell region
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Convert the mask to RGB for display
rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)


# In[10]:


# Specify the path to the main folder
image_folder_path = 'C:\\Users\\FARAH\\Desktop\\Digtial image processing\\Dataset_Pre\\output_pre'
ground_truth_folder_path = r'C:\Users\FARAH\Desktop\Digtial image processing\Segmentation_Ground_Truth\output'


# In[72]:


# Loop through each folder in the main folder
for folder_name in os.listdir(image_folder_path):
    # Create the full path to the current folder in image
    folder_path = os.path.join(image_folder_path, folder_name)
    print(f"Processing folder: {folder_name}")
    # Create the full path to the current folder in ground
    ground_truth_mask_path = os.path.join(ground_truth_folder_path, folder_name)

    # Loop through each file in the current folder
    for file_name in os.listdir(folder_path):
        # Create the full path to the current file in image
        file_path = os.path.join(folder_path, file_name)
        # Create the full path to the current file in ground
        file_path_ground = os.path.join(ground_truth_mask_path, file_name)

        # Read the image
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # Check if the image is loaded successfully
        if image is not None:
            # Convert image data to float
            image_float = image.astype(float) / 255.0  # Normalize pixel values to the range [0, 1]

            # Display the original image
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')

            # Convert the image to the HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds of the purple color in the HSV color space
            lower_bound = np.array([129, 40, 0])
            upper_bound = np.array([174, 255, 255])

            # Create the mask
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Extract the purple pixels using the mask
            purple = cv2.bitwise_and(image, image, mask=mask)

            # Remove noise
            purple = cv2.medianBlur(purple, 5)

            # Convert to grayscale and apply thresholding
            gray = cv2.cvtColor(purple, cv2.COLOR_BGR2GRAY)

            # Threshold the image to create a topographic map
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 6))
            # Opening, Closing
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
            # Erosion
            eroded = cv2.erode(closed, kernel, iterations=2)
            # Dilation
            dilated = cv2.dilate(eroded, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area
            min_area = 100  # Minimum area threshold for white blood cells
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

            # Create a mask for white blood cell region
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

            # Convert the mask to RGB for display
            rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # Display the processed image
            plt.subplot(1, 2, 2)
            plt.imshow(rgb)
            plt.title('Processed Image')
            plt.show()

            # Load the ground truth mask as a color image
            ground_truth_mask = cv2.imread(file_path_ground, cv2.IMREAD_COLOR)

            # Convert the ground truth mask to grayscale
            ground_truth_mask_gray = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)

            # Your existing IoU calculation code...
            rgb1 = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            # Calculate the intersection and union
            intersection = np.logical_and(rgb1, ground_truth_mask_gray)
            union = np.logical_or(rgb1, ground_truth_mask_gray)

            # Calculate the IoU score
            iou = np.sum(intersection) / np.sum(union)
            iou_final.append(iou)


# In[87]:


from tabulate import tabulate

# Initialize an empty list to store the results
results = []

# Loop through each folder in the main folder
for folder_name in os.listdir(image_folder_path):
    # ... your existing code ...

    # Calculate the IoU score
    iou = np.sum(intersection) / np.sum(union)
    iou_final.append(iou)

    # Append the folder name and IoU score to the results list
    results.append([folder_name, iou])

# Print the results in a table format
table_headers = ["Folder Name", "IoU Score"]
print(tabulate(results, headers=table_headers))


# In[88]:


avg = np.mean(iou_final) * 100
print(avg)


# In[ ]:




