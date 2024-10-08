## # if you have 200 images for example and you want to give batches of 10 each time the sample code will be as bellow.

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Example class names (change this based on your specific model)
class_names = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']

# Path to the image directory and expected image size
image_dir = 'path_to_images'
image_size = (224, 224)  # Resize to match the input size of your model
num_channels = 3  # RGB images have 3 channels

# Initialize an empty list to store images
imgs = []

# Load and preprocess 200 images
for img_name in os.listdir(image_dir)[:200]:  # Limit to 200 images
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path)
    img = img.resize(image_size)  # Resize image
    img = np.array(img)  # Convert to NumPy array
    if img.shape == (*image_size, num_channels):  # Ensure the shape is correct
        imgs.append(img)

# Convert list of images to NumPy array
imgs = np.array(imgs)

# Optional: Normalize pixel values to [0, 1]
imgs = imgs / 255.0

# Load a pre-trained model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Process images in batches of 10 and make predictions
batch_size = 10
num_batches = len(imgs) // batch_size

for batch_index in range(num_batches):
    # Get a batch of 10 images
    batch_imgs = imgs[batch_index * batch_size : (batch_index + 1) * batch_size]

    # Add a batch dimension to images if needed (for single image, this would be done by np.expand_dims)
    # Here we already have a batch, so no need to expand dims

    # Make predictions on the batch
    preds = model.predict(batch_imgs)

    # Decode predictions (for ImageNet models)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)

    # Visualize each image with the predicted label
    for i in range(batch_size):
        plt.imshow(batch_imgs[i])
        plt.title(f"Predicted Label: {decoded_preds[i][0][1]}")  # Show predicted class name
        plt.axis('off')
        plt.show()

# If the number of images is not a multiple of batch_size, handle remaining images
remaining_images = len(imgs) % batch_size
if remaining_images > 0:
    batch_imgs = imgs[num_batches * batch_size:]
    preds = model.predict(batch_imgs)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)

    for i in range(remaining_images):
        plt.imshow(batch_imgs[i])
        plt.title(f"Predicted Label: {decoded_preds[i][0][1]}")
        plt.axis('off')
        plt.show()
