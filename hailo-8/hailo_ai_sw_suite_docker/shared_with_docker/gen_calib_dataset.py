# importing everything needed
from hailo_sdk_client import ClientRunner, InferenceContext

import json
import os

import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import pyplot as plt
from PIL import Image

IMAGE_RESOLUTION = 256
IMAGES_TO_VISUALIZE = 5

# First, we will prepare the calibration set. Resize the images to the correct size and crop them.
from tensorflow.python.eager.context import eager_mode


#def preproc(image, output_height=224, output_width=224, resize_side=256):
def preproc(image, output_height=IMAGE_RESOLUTION, output_width=IMAGE_RESOLUTION, resize_side=256):
    ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)

        return tf.squeeze(cropped_image)


images_path = './images'
images_list = [img_name for img_name in os.listdir(images_path) if
               os.path.splitext(img_name)[1] == '.jpg']

calib_dataset = np.zeros((len(images_list), IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    img_preproc = preproc(img)
    calib_dataset[idx, :, :, :] = img_preproc.numpy()

np.save("calib_dataset_"+str(IMAGE_RESOLUTION)+"x"+str(IMAGE_RESOLUTION)+".npy", calib_dataset)

