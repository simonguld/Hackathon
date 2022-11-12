### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Paths
# Save mnist cnn model to the following path
cnn_mnist_path = 'C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Hackathon'

## Imports:
import numpy as np
import matplotlib.pyplot as plt
import importlib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #  {'0', '1', '2', '3'} = {Show all messages, remove info, remove info and warnings, remove all messages}
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers
from sklearn import datasets, model_selection, metrics
from knn_sg import cross_validation
from PIL import Image

### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

## Todo:
# Kan man nøjes med at bygge en gang og så fitte på forskellige data?
# lav en funktion, så cnn på bekvem vis kan fodre til cross-validation

### MAIN ------------------------------------------------------------------------------------------------------------------------------------

def main():

    # Decide on which datasets to run
    number8x8, numbers28x28, cifar10  = False, False, True

    #PART 1: Build and predict labels for 8x8 grescale images of numbers from 0-9
    if number8x8:
        # Load 1797 samples of 8x8 grescale images (flattened) of handwritten numbers from 0-9 [10 classes].
        digits = datasets.load_digits()
        #print(digit_data['DESCR'])

        sample_points, _ = digits.data.shape
        pixel_shape = [8,8]

        #reshape each image to an 8x8 matrix and normalize
        digit_data = digits.data.reshape(sample_points, pixel_shape[0], pixel_shape[1]) # / 255.0
        digit_labels = digits.target
    
        n_classes = 10
    
        #Split dataset
        train_data, test_data, train_labels, test_labels = \
                            model_selection.train_test_split(digit_data, digit_labels, test_size = 0.2, random_state = 25)

        # Build CNN model
        digit_model = models.Sequential()

        #1st convolution
        #digit_model.add(layers.Conv2D())
        digit_model.add(layers.Conv2D(24,(3,3), activation = 'relu', input_shape = (8,8,1)))
        digit_model.add(layers.MaxPool2D((2,2)))

        digit_model.add(layers.Conv2D(64,(2,2), activation = 'relu'))
        #digit_model.add(layers.MaxPool2D((2,2)))

        digit_model.add(layers.Flatten())
        digit_model.add(layers.Dense(36, activation = 'relu'))
        digit_model.add(layers.Dense(n_classes))

        digit_model.summary()


        # Compile
        """
        Model.compile(
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs
    )
        """

        digit_model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['acc'])

        # Train model
        history = digit_model.fit(train_data, train_labels, epochs=10, 
                        validation_data=(test_data, test_labels))


        y_pred = digit_model.predict(test_data, batch_size = 32, verbose = 1)
        y_pred_bool = np.argmax(y_pred, axis = 1)

        print(metrics.classification_report(test_labels, y_pred_bool))

    if numbers28x28:


        #60000 28x28 grescale handwritten numbers from 0-9
        # Load data
        (data_train, labels_train), (data_test, labels_test) = keras.datasets.mnist.load_data( )

        ## Preprocessing 
        # Normalize
        data_train = data_train / 255
        data_test = data_test / 255

        pixel_shape = np.empty(2)
        samples, pixel_shape[0], pixel_shape[1] = data_train.shape


        build_and_train, load_model = False, True

        ## Build and train
        if build_and_train:
            #build model
            model = models.Sequential()

            #Build
            model.add(layers.Conv2D(16,(4,4), activation = 'relu', input_shape = (28,28,1)))
            model.add(layers.MaxPool2D(2,2))

            model.add(layers.Conv2D(16,(3,3), activation = 'relu'))
            model.add(layers.MaxPool2D(2,2))

            model.add(layers.Conv2D(16,(2,2), activation = 'relu'))
            model.add(layers.MaxPool2D(2,2))

            model.add(layers.Flatten())

            model.add(layers.Dense(16, activation = 'relu'))
            model.add(layers.Dense(10))

            model.summary()

            #Compile
            model.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['acc'])


            #Train
            history = model.fit(data_train,labels_train, epochs = 4, verbose = 1, validation_data=(data_test,labels_test))
            
            #Save
            model.save(cnn_mnist_path)

        ## If already trained, load model
        if load_model:
            model = keras.models.load_model(cnn_mnist_path)

        
        # 'Cross validate' (not really a proper cross validation by testing on traning data on which the model was built)
        x_train, x_test, x_labels, x_test_labels = model_selection.train_test_split(data_train, labels_train, test_size = 0.2, \
            random_state = 25)
        
        y_pred = model.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis = 1)

        print(metrics.classification_report(x_test_labels, y_pred_bool))

    if cifar10:
        #cifar contains 50000 training images and 10000 test images with dimension 32x32x3 distributed over 10 classes

        (data_train, labels_train), (data_test, labels_test) = keras.datasets.cifar10.load_data( )
        
       
        ## Preprocessing:
        pixel_shape = np.empty(3)
        train_samples, pixel_shape[0], pixel_shape[1], pixel_shape[2] = data_train.shape

        # Ensure that all data has been loaded correctly
        assert data_train.shape == (50000, 32, 32,3)
        assert data_test.shape == (10000, 32, 32,3)
        assert labels_train.shape == (50000,1)
        assert labels_test.shape == (10000,1)

        # Normalize
        data_train = data_train / 255
        data_test = data_test / 255

        ## Building and traning model

        model = models.Sequential()

        model.add(layers.Conv2D(128,(3,3), activation = 'relu', input_shape = (32,32,3)))
        model.add(layers.MaxPool2D(2,2))

        model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
        model.add(layers.MaxPool2D(2,2))

        model.add(layers.Conv2D(32,(3,3), activation = 'relu'))
        model.add(layers.MaxPool2D(2,2))


        model.add(layers.Flatten())

        model.add(layers.Dense(32, activation = 'relu'))
        model.add(layers.Dense(10))

        model.summary()

        #Compile
        optimizer_primary = 'adam'
        optimizer_alternative = optimizers.RMSprop()
        model.compile(optimizer=optimizer_primary, 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['acc'])
        # optimizer   optimizer=keras.optimizers.RMSprop()

        #Train
        history = model.fit(data_train,labels_train, epochs = 15, verbose = 1, batch_size = 32, \
                         validation_data=(data_test,labels_test))
        # batch size 64
        if 0:
            #Save
            model.save(cnn_mnist_path)

        ## If already trained, load model
        if 0:
            model = keras.models.load_model(cnn_mnist_path)





if __name__ == '__main__':
    main()
 #Build
        