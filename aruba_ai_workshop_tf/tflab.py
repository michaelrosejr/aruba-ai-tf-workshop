import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = tf.keras.utils.image_dataset_from_directory('data', label_mode='int')


data_iterator = data.as_numpy_iterator()  # type: ignore
batch = data_iterator.next()
# 0 is access points
# 1 is switches

# print(batch[0].shape)
# print(batch[1])

#
# Show images from batch
#
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(np.uint8))
#     ax[idx].title.set_text(batch[1][idx])

# plt.show()

####
# SCALE DATA
#
data = data.map(lambda x, y: (x/255, y))  # type: ignore
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(batch[1][idx])

# plt.show()

####
# SPLIT DATA
#
# print(len(data))
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

# print(train_size+val_size+test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# print(len(test))

from tensorflow.keras.layers import (Conv2D, Dense, Flatten,  # type: ignore
                                     MaxPooling2D)
####
# Build Deep Learning Model
#
from tensorflow.keras.models import Sequential  # type: ignore

model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# print(model.summary())

####
# Train Model
#
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

####
# Plot Performance
#
##
# Loss
#
# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='loss')
# plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
# fig.suptitle('Loss', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()

##
# Accuracy
#
# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
# plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
# fig.suptitle('Accuracy', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()

####
# Evaluate Model
#
# from tensorflow.keras.metrics import BinaryAccuracy  # type: ignore
# from tensorflow.keras.metrics import Precision, Recall

# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()

# for batch in test.as_numpy_iterator():
#     X, y = batch
#     yhat = model.predict(X)
#     pre.update_state(y, yhat)
#     re.update_state(y, yhat)
#     acc.update_state(y, yhat)

# print(pre.result(), re.result(), acc.result())

####
# Test Model
#
# import cv2

# img = cv2.imread('test_images/test_switch-1.png')
# resize = tf.image.resize(img, (256, 256))
# plt.imshow(resize.numpy().astype(int))  # type: ignore
# plt.show()

# yhat = model.predict(np.expand_dims(resize/255, 0))  # type: ignore

# if yhat > 0.5:
#     print('Predicted class is an Access Point')
# else:
#     print('Predicted class is a Switch')


####
# Save Model
#
model.save(os.path.join('models', 'imageclassifier.h5'))
