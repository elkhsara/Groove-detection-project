import numpy as np 
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import load_model
from skimage.segmentation import find_boundaries
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Model
from skimage.morphology import binary_closing
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from sklearn.preprocessing import StandardScaler
from keras.losses import mean_squared_error, binary_crossentropy
from skimage.morphology import binary_erosion, binary_dilation, disk


def get_data(images_folder, labels_folder):
    print(f"Getting images from {images_folder}")
    print(f"Getting labels from {labels_folder}")
    image_files = os.listdir(images_folder)
    images = []
    labels = []
    nan_count = 0

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        lbl_path = os.path.join(labels_folder, img_file)
        img = np.load(img_path)
    
        if np.isnan(img).any():
            nan_count += 1
        else:
            lbl = np.load(lbl_path)
            images.append(img)
            labels.append(lbl)
    print("Loading data done")
    return images,labels

def get_test_images(images_folder):
    print(f"Getting images from {images_folder}")
    image_files = os.listdir(images_folder)
    images = []
    nan_count = 0
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = np.load(img_path)
    
        if np.isnan(img).any():
            nan_count += 1
        else:
            images.append(img)
    print("Loading data done")
    return images

def count_zero_images(images):
    num_zero_images = 0
    for image in images:
        if np.all(image == 0):
            num_zero_images += 1
    return num_zero_images

def process_data_for_model(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    # Add a new axis to images and labels
    images = images[:, :, :, np.newaxis]
    labels = labels[:, :, :, np.newaxis]
    return images, labels

def predict_image(model, image):
    image = np.array(image)
    image = image[np.newaxis, :, :, np.newaxis]
    predictions = model.predict(image)
    binary_predictions = (predictions > 0.5).astype(np.uint8)
    binary_predictions = binary_predictions.squeeze()
    
    return binary_predictions

# Define metrics functions
def intersection_over_union(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    iou = intersection / union
    return iou

def pixel_accuracy(y_true, y_pred):
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    accuracy = correct_pixels / total_pixels
    return accuracy

def dice_coefficient_metric(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2. * intersection) / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return dice

def boundary_f1_score(y_true, y_pred):
    # Assuming y_true and y_pred are binary masks
    boundary_true = find_boundaries(y_true)
    boundary_pred = find_boundaries(y_pred)
    
    tp = np.sum(np.logical_and(boundary_true, boundary_pred))
    fp = np.sum(np.logical_and(np.logical_not(boundary_true), boundary_pred))
    fn = np.sum(np.logical_and(boundary_true, np.logical_not(boundary_pred)))
    
    precision = tp / (tp + fp + 1e-6) 
    recall = tp / (tp + fn + 1e-6)  
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  
    return f1

def preprocess_images(images):
    preprocessed_images = []
    scaler = StandardScaler()
    for img in images:
        normalized_img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
        preprocessed_images.append(normalized_img)
    return np.array(preprocessed_images)

def augment_images(images, labels, shift_range=0.1):
    augmented_images = []
    augmented_labels = []
    
    for img, lbl in zip(images, labels):
        # Append the initial image and label
        augmented_images.append(img)
        augmented_labels.append(lbl)

        # Flip image and label horizontally
        flipped_img = np.fliplr(img)
        flipped_lbl = np.fliplr(lbl)
        augmented_images.append(flipped_img)
        augmented_labels.append(flipped_lbl)

        # Shift image and label horizontally
        shift = np.random.uniform(0, shift_range) * img.shape[1]
        shifted_img = np.roll(img, int(shift), axis=1)
        shifted_lbl = np.roll(lbl, int(shift), axis=1)
        augmented_images.append(shifted_img)
        augmented_labels.append(shifted_lbl)
    
    return np.array(augmented_images), np.array(augmented_labels)

def improve_labels(masks,n=3):
    processed_masks = []
    for mask in masks:
        # Apply binary closing operation to fill small holes and gaps
        binary_mask = mask.astype(bool)
        mask = binary_closing(binary_mask, disk(n))
        processed_masks.append(mask)
    return np.array(processed_masks)


def inverse_sigmoid(x):
    return np.log(x / (1 - x))

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
