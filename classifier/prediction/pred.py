import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

class_labels=['T-Shirt/Top','Trousors','Pullover','Dress','Coat','Slippers/Sandal','Shirt','Shoe/Sneaker','Bag','Ankle Boot']

loaded_model=load_model('./classifier/prediction/Test_model3')

#tf.global_variables_initializer()
global graph
graph = tf.get_default_graph() 


def db_predict_image(img):
    pred = dict()
    
    path='./classifier/prediction/db_images/'
    img_array = cv2.imread(path+img ,cv2.IMREAD_GRAYSCALE) 
    IMG_SIZE = 28
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=tf.keras.utils.normalize(new_array,axis=1)
    new_array=new_array.reshape(1,28,28,1)
    with graph.as_default():
        prediction=loaded_model.predict(new_array)
    pred['prediction_class'] = class_labels[np.argmax(prediction)]
    pred['prediction_perc'] = np.around(np.amax(prediction)*100)
    return pred

def user_predict_image(img):
    path='./classifier/prediction/user_images/'
    img_array = cv2.imread(path+img ,cv2.IMREAD_GRAYSCALE) 
    IMG_SIZE = 28
    pred=dict()
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=tf.keras.utils.normalize(new_array,axis=1)
    new_array=new_array.reshape(1,28,28,1)
    with graph.as_default():
        prediction=loaded_model.predict(new_array)
    pred['prediction_class'] = class_labels[np.argmax(prediction)]
    pred['prediction_perc'] = np.around(np.amax(prediction)*100)
    return pred

def user_image(img):
    path='./classifier/prediction/user_images/'
    img_array = cv2.imread(path+img ,cv2.IMREAD_GRAYSCALE) 
    IMG_SIZE = 28
    pred=dict()
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=tf.keras.utils.normalize(new_array,axis=1)
    new_array=new_array.reshape(1,28,28,1)
    with graph.as_default():
        prediction=loaded_model.predict(new_array)
    pred['prediction_class'] = class_labels[np.argmax(prediction)]
    pred['image'] = img
    return pred