### SETUP -----------------------------------------------------------------------------------------------------------------------------------




## Imports:
import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #  {'0', '1', '2', '3'} = {Show all messages, remove info, remove info and warnings, remove all messages}
import tensorflow as tf

from tensorflow import keras
from keras import models, layers, optimizers
from sklearn import datasets, model_selection, metrics
from PIL import Image

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
cnn_mnist_path = './cnn_mnist.h5'


### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def cross_validation(x_train, train_labels, method, test_fraction = 0.2, f1_score_average = 'weighted'):
    """
    Randomly shuffles and splits a training set into a traning and test set, whose relative size is determined by test_fraction.
    If the training set is divided into N sets, one will act as traing set and the remaning N-1 sets will act as training data.
    The function continues until all N sets has acted as training set and returns an f1 score for each roll.

    params:
            x_train: Training data. An nd.array of dimension (No. of points, dimension of points). 
            train_labels: Traning data labels. An nd.array of dimension (No. of points, 1)
            method: the machine learning classification method used to assign labels. When used as parameter to the function,
            it must be restricted to only depend on (x_train,train_labels,x_test) in that order
            test_fraction: The approximate fraction of training data that will act as test data in a given roll
    returns:
            a list of f1-scores for each roll
    """
  

    # find number of training points
    n_points, _ = x_train.shape
    # find index to split data set
    split_index = np.floor( n_points * test_fraction)
    # find number of cross validation subsets / rolls
    rolls = int(np.floor(n_points / (split_index)))

    ## shuffle training data and corresponding labels
    # generate random integer values from [0, no. of points]
    row_shuffle = np.linspace(0, n_points - 1, n_points).astype('int')
    random_generator = np.random.default_rng()
    random_generator.shuffle(row_shuffle)

    # Apply random shuffling to training data and labels
    shuffled_data = x_train.astype('float')[row_shuffle]
    shuffled_labels = train_labels.astype('int')[row_shuffle]

    # Make list for storing f1 scores
    f1_list = []

    # Perform the roll, letting each subset act as test data
    for i in range(rolls):
        # The rows that will act as test data in current roll
        index_range_test = [int (i * split_index),  int ((i + 1) * split_index)]

        # Extract test_data and corresponding true labels for current roll
        test_data = shuffled_data[index_range_test[0]:index_range_test[1]]
        test_labels_true = shuffled_labels[index_range_test[0]:index_range_test[1]]

        # Extract training data and corresponding true labels for current roll
        train_data = np.delete(shuffled_data, np.arange(index_range_test[0], index_range_test[1]), axis = 0)
        train_labels = np.delete(shuffled_labels, np.arange(index_range_test[0], index_range_test[1]), axis = 0)

        # Assign test point labels using the provided method
        test_labels = method(train_data, train_labels, test_data)

        # Calculate and store the f1-score of current roll
        f1_score = metrics.f1_score(test_labels_true, test_labels, average = f1_score_average)
        f1_list.append(f1_score)

    return f1_list

## Todo:
# Kan man nøjes med at bygge en gang og så fitte på forskellige data?
# lav en funktion, så cnn på bekvem vis kan fodre til cross-validation

### MAIN ------------------------------------------------------------------------------------------------------------------------------------

def main():

    # Decide on which datasets to run
    number8x8, numbers28x28, cifar10  = False, True, False

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

        t_start = time.time()
        # Train model
        history = digit_model.fit(train_data, train_labels, epochs=10, 
                        validation_data=(test_data, test_labels))


        y_pred = digit_model.predict(test_data, batch_size = 32, verbose = 1)
        y_pred_bool = np.argmax(y_pred, axis = 1)

        print(metrics.classification_report(test_labels, y_pred_bool))
        t_end = time.time()
        print('Time elapsed: ', t_end - t_start)
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

        print(data_train.shape, data_test.shape)

        build_and_train, load_model = True, False
        t_start = time.time()
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
            history = model.fit(data_train,labels_train, epochs = 1, verbose = 1, validation_data=(data_test,labels_test))
            
            #Save
            model.save(cnn_mnist_path)

        ## If already trained, load model
        if load_model:
            model = keras.models.load_model(cnn_mnist_path)

        t_end = time.time()
        print('Time elapsed: ', t_end - t_start)
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
        assert data_train.shape == (50000, 32, 32, 3)
        assert data_test.shape == (10000, 32, 32, 3)
        assert labels_train.shape == (50000, 1)
        assert labels_test.shape == (10000, 1)

        # Normalize
        data_train = data_train / 255
        data_test = data_test / 255

        ## Building and traning model

        ## lege med: 
        # preprocessing --> udvide træningssæt med rotationer, zoom ins, brightness adj.,
        # --> OBS: keras.layers har nogle funktioner.
        # forskyde data så av = 0
        # Max-pool vs conv2d inden connected layer
        # optimizer & loss function
        # 3D conv og maxpool lag
        # global max pool
        # batch size
        # trying on a softmax activation for class neurons
        # fine tuning with random search
        # including regularization l2. regulariser indtil minimering af validation loss
        # including dropout layers
        # Stacking conv2d layers
        # batch normalization
        # tilføje koefficienter indtil grænen for overfitting nås. [minier validation loss]

        model = models.Sequential()

        model.add(layers.Conv2D(64,(4,4), activation = 'relu', input_shape = (32,32,3)))
        model.add(layers.MaxPool2D(2,2))

        model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
        model.add(layers.MaxPool2D(2,2))

        model.add(layers.Conv2D(96,(3,3), activation = 'relu'))
        model.add(layers.MaxPool2D(2,2))

       # model.add(layers.Conv2D(128,(2,2), activation = 'relu'))
        #model.add(layers.Conv2D(16,(3,3), activation = 'relu'))
        #model.add(layers.MaxPool2D(2,2))

        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation = 'relu'))
        model.add(layers.Dense(10), activation = 'softmax')

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
        