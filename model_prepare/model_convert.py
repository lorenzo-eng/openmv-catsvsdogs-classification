import tensorflow as tf
import tensorflow_datasets as tfds
import config

(raw_test, ), metadata = tfds.load(
    'cats_vs_dogs',
    split=config.CONVERT_SPLIT,
    with_info=True,
    as_supervised=True,
)

test = raw_test.map(config.img_to_model_input)
test_batches = test.batch(1)

def representative_data_gen():
    for input_value, _ in test_batches.take(100):
        yield [input_value]

model = tf.keras.models.load_model(config.KERAS_MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
open(config.TFLITE_MODEL_PATH, "wb").write(tflite_model)
