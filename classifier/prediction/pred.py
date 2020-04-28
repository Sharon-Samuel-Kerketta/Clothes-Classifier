import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
from tensorflow.python.keras.backend import set_session

class_labels=['T-Shirt/Top','Trousors','Pullover','Dress','Coat','Slippers/Sandal','Shirt','Shoe/Sneaker','Bag','Ankle Boot']

global graph,sess
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

set_session(sess)
loaded_model=load_model('./classifier/prediction/model92_with_aug.h5')

def db_predict_image(img):
    pred = dict()
    IMG_SIZE = 28
    path='./classifier/prediction/db_images/'
    
    img_array = cv2.imread(path+img).astype('float32')
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=new_array.reshape(1,28,28,1)
    new_array = new_array.astype('float32')
    new_array /= 255.0

    with graph.as_default():
        set_session(sess)
        prediction=loaded_model.predict(new_array)
    pred['prediction_class'] = class_labels[np.argmax(prediction)]
    pred['prediction_perc'] = np.around(np.amax(prediction)*100)
    return pred

def user_predict_image(img):
    path='./classifier/prediction/user_images/'
    IMG_SIZE = 28
    pred=dict()

    img_array = cv2.imread(path+img).astype('float32')
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=new_array.reshape(1,28,28,1)
    new_array = new_array.astype('float32')
    new_array /= 255.0

    with graph.as_default():
        set_session(sess)
        prediction=loaded_model.predict(new_array)
    pred['prediction_class'] = class_labels[np.argmax(prediction)]
    pred['prediction_perc'] = np.around(np.amax(prediction)*100)
    return pred

def user_image(img):
    path='./classifier/prediction/user_images/'
    IMG_SIZE = 28
    pred=dict()

    img_array = cv2.imread(path+img).astype('float32')
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=new_array.reshape(1,28,28,1)
    new_array = new_array.astype('float32')
    new_array /= 255.0

    with graph.as_default():
        set_session(sess)
        prediction=loaded_model.predict(new_array)
    pred['prediction_class'] = class_labels[np.argmax(prediction)]
    pred['image'] = img
    return pred