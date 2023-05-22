import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore

img = cv2.imread('test_images/test_switch-1.png')
resize = tf.image.resize(img, (256, 256))

model = load_model('models/imageclassifier.h5')
model_array = model.predict(np.expand_dims(resize/255, 0))  # type: ignore
print(model_array)

yhat = model.predict(np.expand_dims(resize/255, 0))  # type: ignore

if yhat > 0.5:
    print('Predicted class of image is a Switch')
else:
    print('Predicted class of image is an Access Point')
