import io

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

#CONSTANTS
DATA_DIR = "data/"
TRAIN_DIR = "training/"
VAL_DIR = "validation/"
TEST_DIR = "test_set_images/"

def dataset_generator_from_files(paths, one_hot=True):
    for image_path, mask_path in paths:
        image, mask = tf.io.read_file(image_path), tf.io.read_file(mask_path)
        image, mask = tf.io.decode_png(image), tf.io.decode_png(mask)
        
        if one_hot:
            mask = tf.squeeze(mask)
            mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
            mask = tf.math.round(mask)
            mask = tf.cast(mask, dtype=tf.int32)
            mask = tf.one_hot(mask, depth=2)
        
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
        yield image, mask
    

def augmentation(image, mask, crop_size=128, rotate=True, flip=True, mean_filter=False):
    
    x = tf.random.uniform(shape=[], maxval=image.shape[1]-crop_size, dtype=tf.int32)
    y = tf.random.uniform(shape=[], maxval=image.shape[1]-crop_size, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, x, y, crop_size, crop_size)
    mask = tf.image.crop_to_bounding_box(mask, x, y, crop_size, crop_size)
    
    if rotate:
        k = tf.random.uniform(shape=[], maxval=4, dtype=tf.int32)
        image, mask = tf.image.rot90(image, k=k), tf.image.rot90(mask, k=k)
    
    if flip:
        p = tf.random.uniform(shape=[])
        if p <0.5:
            image, mask = tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
    
    if mean_filter:
        image = tfa.image.mean_filter2d(image)
    
    return image, mask


def scheduler(epoch, lr):
    power = tf.cast(epoch // 1000, dtype=tf.float32)
    return lr * tf.math.exp(-power)