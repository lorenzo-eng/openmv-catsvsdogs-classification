import os
import tensorflow as tf

IMG_SIZE = 128 # images will be resized to 128x128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def img_to_model_input(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5)-1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def model_input_to_plot(image):
    return (image+1)/2

# train params

WIDTH_MULTIPLIER = 0.25
EPOCHS = 20
BATCH_SIZE = 32
TRAIN_SPLIT = ['train[:10%]', 'train[10%:20%]', 'train[20%:30%]']

# convert params

CONVERT_SPLIT = ['train[30%:40%]']

# test params

TEST_IMAGES = 25
TEST_ROWS = 5
TEST_COLS = 5
TEST_SPLIT = ['train[40%:45%]']

OUTPUT_DIR = 'build'
MODEL_FILEN = 'catsvsdogs_model'
TFLITE_MODEL_FILE = MODEL_FILEN + '.tflite'
TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, TFLITE_MODEL_FILE)
KERAS_MODEL_FILE = MODEL_FILEN + '.h5'
KERAS_MODEL_DIR = 'keras'
KERAS_MODEL_PATH = os.path.join(OUTPUT_DIR, KERAS_MODEL_DIR, KERAS_MODEL_FILE)
