import tensorflow as tf 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils_gradcam as utils

val_imgs = 'val_1000'
class_info = {0: 'Cat', 1: 'Dog'}

x, y, files = utils.manual_pre_process(val_imgs, 224)

img = x[59]
label = y[59]
path = files[59]

model = tf.keras.models.load_model('mobilenetv2_epochs25_batch25_sample_size2000_aug_0.h5', custom_objects={"precision_m": utils.precision, "recall_m": utils.recall})

pred_raw = model.predict(np.expand_dims(img, axis=0))[0][0]
pred = utils.decode_prediction(pred_raw)
pred_label = class_info[pred]

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(pred_label + ' ' + str(pred_raw))