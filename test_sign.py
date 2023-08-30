import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

img = cv2.imread('images/7142491.png')

resize = tf.image.resize(img, (256,256))

model = load_model('models/imageclassifier_TIN.h5')

yhat = model.predict(np.expand_dims(resize/255, 0))

print(yhat)

if yhat >0.5 :
    print('NID Back Card')
else:
    print('Others')