import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import config


(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=config.TRAIN_SPLIT,
    with_info=True,
    as_supervised=True,
)

train_elements = 0
for element in raw_train:
  train_elements += 1

train = raw_train.map(config.img_to_model_input)
validation = raw_validation.map(config.img_to_model_input)
test = raw_test.map(config.img_to_model_input)


train_batches = train.shuffle(train_elements + 1).batch(config.BATCH_SIZE)

validation_batches = validation.batch(config.BATCH_SIZE)
test_batches = test.batch(config.BATCH_SIZE)

# Create the base model from the pre-trained model MobileNet V1
base_model = tf.keras.applications.MobileNet(input_shape=config.IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               alpha=config.WIDTH_MULTIPLIER)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# derive a probability distribution for the available classes
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

print(model.summary())

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches,
                    epochs=config.EPOCHS,
                    validation_data=validation_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.savefig(os.path.join(config.OUTPUT_DIR, "training_out.png"))
plt.show()

model.save(config.KERAS_MODEL_PATH)
