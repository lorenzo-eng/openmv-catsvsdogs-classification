import sys
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import tensorflow_datasets as tfds
import config


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    ax = axes[i//config.TEST_ROWS][i%config.TEST_COLS]

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    img = np.squeeze(img)
    ax.imshow(config.model_input_to_plot(img), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    ax.set_xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)

(raw_test, ), metadata = tfds.load(
    'cats_vs_dogs',
    split=config.TEST_SPLIT,
    with_info=True,
    as_supervised=True,
)

test = raw_test.map(config.img_to_model_input)
test_batches = test.batch(1)

interpreter = tf.lite.Interpreter(model_path=config.TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels, test_imgs = [], []

for img, label in test_batches.take(config.TEST_IMAGES):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))

    test_labels.append(label.numpy()[0])
    test_imgs.append(img)

class_names = ['cat','dog']

fig, axes = plt.subplots(config.TEST_ROWS,config.TEST_COLS, figsize=(config.TEST_ROWS*2,config.TEST_COLS*2))
for index in range(config.TEST_IMAGES):
    plot_image(index,predictions,test_labels,test_imgs)

plt.savefig(os.path.join(config.OUTPUT_DIR, "test_out.png"))
plt.show()
