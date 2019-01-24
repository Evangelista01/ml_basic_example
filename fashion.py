# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:24:16 2018

@author: Evangelista
"""

from keras.datasets import fashion_mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def transfer_name(number):
    if number == 0:
        return "T-shirt/top"
    if number == 1:
        return "Trouser"
    if number == 2:
        return "Pullover"
    if number == 3:
        return "Dress"
    if number == 4:
        return "Coat"
    if number == 5:
        return "Sandal"
    if number == 6:
        return "Shirt"
    if number == 7:
        return "Sneaker"
    if number == 8: 
        return "Bag"
    if number == 9:
        return "Ankle boot"
    
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    """building a sequential type of model"""
    model = tf.keras.models.Sequential()
    #not to have multi-dim array so we flat it to line in input layer
    model.add(tf.keras.layers.Flatten())
    #now hidden layer
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.sigmoid)) # 128 neurons in the layer, activation function
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.sigmoid))
    #output layer
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #10 in number of classifications 10 - because numbersof 0-9 in the data we want to distinguish
    
    
    """parameters for training of the model"""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    """train the model"""
    model.fit(x_train, y_train, epochs=5)
    
    """calculate validation loss and validation accucary"""
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)
    
    """saving and loading a model"""
    #model.save_weights('epic_num_reader.model.h5')
    #model.save('epic_num_reader.model')
    #new_model = tf.keras.models.load_model('epic_num_reader.model')
    
    """predictions"""
    predictions = model.predict([x_test])
    #print(predictions) #weights of every test sample and every numbe
    for i in range(0, 20):
        print("==========================================")
        print("IMAGE: ")
        plt.imshow(x_test[i], cmap = plt.cm.binary)
        plt.show()
        print("PREDICTION: " + transfer_name(np.argmax(predictions[i])))
        print("TRUE: " + transfer_name(y_test[i]))
        print("==========================================")