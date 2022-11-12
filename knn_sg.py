### SETUP -------------------------------------------------------------------------------------------------------------------------------------

## Imports
import time
import sys
import numpy as np
from scipy import spatial 
from sklearn import metrics, datasets, naive_bayes

## Paths
# Paths for traning data, training labels and test data and test labels
path_train = "C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Hackathon\\dataX.dat"
path_train_labels = "C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Hackathon\\dataY.dat"
path_test = "C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Hackathon\\dataXtest.dat"
path_test_labels = "C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Hackathon\\dataYtest.dat"

## Print out setting: 
# if true, a file 'dataY_test.txt' holding the labels of the test data will be printed out to the 'path_test_labels' path
generate_label_file = False

### FUNCTIONS --------------------------------------------------------------------------------------------------------------------------------

def gaussian_naive_bayes(x_train, train_labels, x_test, runtime_information = False):
    """
    Predict the labels of a test data set given the labels of a traning data set using Gaussian Naive Bayes.
    If runtime_information = True, the Naive Bayes fitting runtime will be printed.
    Returns an array of test data labels along with an object of the GaussianNB class. 
    For documentation and further information, see
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
    """
    # acces a member of the GaussianNB class and use it to predict labels
    gnb = naive_bayes.GaussianNB()

    test_labels = gnb.fit(x_train, train_labels).predict(x_test)

    return test_labels, gnb
def k_nearest_neighbors(x_train, train_labels, x_test, leaf_size, nearest_neighbors, runtime_information = True, binary_labels = False, \
                        labeling_warning = True):
    """
    Find the labels of a test data set using kd-trees and the method of k-nearest neighbours

    params:
            x_train: Training data. An nd.array of dimension (No. of training points, dimension of points). 
            train_labels: Traning data labels. An nd.array of dimension (No. of points, 1)
            x_test: Test data. An nd.array of dimension (No. of test points, dimension of points)
            leaf_size: The number of data points at which the algorithm resorts to brute force calculation of
                        distances
            nearest_neighbors: The number of nearest point used to deciding the label of a test point
            runtime_information: If true, the time it takes to build the Kd-tree and find k-nearest neighbors for all points is printed
            binary_labels: If true, the function assumes only two classes and does not check whether a label can be uniquely assigned
            labeing_warning: If true, the function will print out when a label cannot be uniquely assigned (can only happen for more than
            2 classes). If, for instance, a test point has an equal number of nearest neighbors belonging to more than one class, a warning
            will be printed
    returns:
            test_labels: A matrix holding the assigned labels to the test points
            If generate_label_file is set to True in the SETUP section, a txt file with all test labels will be created in the path specified
            also in the SETUP section

    """


     ## build kdd tree and find nearest neighbors

    #record build time
    time_init = time.time()

    #build tree
    kd_tree = spatial.KDTree(x_train,leafsize = leaf_size)

    time_end = time.time() - time_init

    if runtime_information:
        print(f'Kd-tree buildtime: ', time_end)

    # time nn search runtime for different leafsizes
    time_init = time.time()

    # find nearest neighbors and their row indices using the Manhattan norm
    nn, index = kd_tree.query(x_test,k = nearest_neighbors, p = 1)

    ## assign labels to test data
    test_labels = assign_label(index, train_labels, binary_labels, labeling_warning)
    
    time_end = time.time() - time_init
    if runtime_information:
        print(f'NN-assignment runtime: ', time_end)


    if generate_label_file:
        # generate a txt file with labels for the test file 'test_labels.txt'
        #Redirect output stream to 'dataYtest.txt'
        orig_output = sys.stdout
        with open(path_test_labels, 'w') as f:
            sys.stdout = f
            print(test_labels)

        #Redirect output stream to terminal
        sys.stdout = orig_output

    return test_labels
def assign_label(nn_index, train_labels, binary_labels = False, labeling_warning = True):
    """
    Finds the labels of test points in accordance with the knn majority voting scheme. A test point is assigned the same label
    as the majority of its k nearest neighbors

    params:
            nn_index: An (N_test_points) x (k nearest neighbors) matrix holding the indices of the k nearest training points \
                for each test point
            train_labels: An (N_training_points) x 1 matrix holding the labels of all training points
            binary_labels: Default = False. If True, each test point can be assigned a unique label when using an uneven no.
                           of nearest neighbors. If false, the function make the user aware if some points cannot be assigned a unique label.
                           In such a case, the test point will simply be assigned the label of smallest value.
    returns: An (N_test_points) x 1 matrix holding all test points labels

    """

    # Construct an array holding all possible label values
    labels_arr = np.unique(train_labels).astype('int')
    # Hold biggest label value
    max_label = np.max(labels_arr)

    if binary_labels == False:
        # If there a more than two classes of labels, we must ensure that each test point is closer to more points of a given class 
        # than any other

        # For each test point, count the number of times each label occurs
        # test label_bins is an (N_test_points) x (no. of labels) matrix counting the occurence of each label for each point
        test_label_bins = np.apply_along_axis(lambda x: np.bincount(x, minlength = max_label+2), \
                            axis = 1, arr = train_labels[nn_index].astype('int'))

  
        # find the maximum number of label occurences for each point
        max_occurrence = np.max ( test_label_bins, axis = 1)

        # check whether different labels occur the same number of times
        # Record no. of test points
        n_test_points, _ = nn_index.shape

        # Initialize a list of test points that cannot be assigned a unique label
        index_multiplicity = []
        for i in range(n_test_points):
            # check if one label occurs more than all others
            label_multiplicity = np.sum(test_label_bins[i] == max_occurrence[i])
            if label_multiplicity > 1:
                index_multiplicity.append(i)

        # Print out a warning if ambiguous labeling has occured
        if labeling_warning and len(index_multiplicity) > 0:
            print(f'The following test points, referenced by their indices with respect to the file holding them, \n \
cannot be assigned a unique label and has simply been assigned the smallest label. \n \
Try altering the nearest neighbor parameter. ')
            print(index_multiplicity)

    # calculate the labels of the test points and return. 
    test_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis = 1, arr = train_labels[nn_index].astype('int'))
    return test_labels
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
  
### MAIN --------------------------------------------------------------------------------------------------------------------------------------


def main():
    knn_leaf_size, cross_validate, naive_bayes, try_new_data = False, False, False, True

    # load test data, training data and training labels
    x_test = np.loadtxt(path_test)
    x_train = np.loadtxt(path_train)
    train_labels = np.loadtxt(path_train_labels)
    np.place(train_labels, train_labels == -1, 0)

    # extract no. of training and test points, along with the dimension of the points
    n_train, dim = x_train.shape
    n_test, _ = x_test.shape

    # set nearest neighbor parameters
    nearest_neighbors = 3



     ## PART 1: Build kd-tree and assign labels to test points using knn
    if knn_leaf_size:
       

        # Examine the tree build time and label assignment runtime for different leaf sizes

        # find leaf size that minimizes build and search time
        leaf_size = [1, 3, 7, 10, 25, 50, 100]

        for leaf in leaf_size:
            print(f'For leaf size {leaf}: ')
            k_nearest_neighbors(x_train, train_labels, x_test, leaf, nearest_neighbors = nearest_neighbors)
 
    ## PART 2: Perform cross validation on training set
    if cross_validate:
        # Compare performance for different leaf sizes
        leaf_size = [1, 3, 7, 10, 25, 50, 100]
 
        for leaf in leaf_size:
            print(f'\nFor leaf size: ', leaf)
            f1_scores = cross_validation(x_train,train_labels, lambda x_train, train_labels, x_test: \
            k_nearest_neighbors(x_train, train_labels, x_test, leaf_size = 10, nearest_neighbors= nearest_neighbors, \
                runtime_information=False), test_fraction = 0.2)
            print("f1 scores for each roll: ", f1_scores, "  Average f1 scores: ", np.average(f1_scores))


    ## PART 3: Get Gaussian Naive Bayes up and running
    if naive_bayes:
        # Try Naive Bayes on x_train data set

        nb_test_labels, gnb = gaussian_naive_bayes(x_train, train_labels, x_test)

        #gnb is a class with attributes like class_count_, classes_, n_features_in_ etc
        #eg print(gnb.class_count_, gnb.classes_)

        # Apply crossvalidation scheme using naive bayes to get an idea of efficay
        # Restrict function to only output the test data labels
        gnb_restricted = lambda data, data_labels, test: gaussian_naive_bayes(data, data_labels, test)[0]
        nv_f1 = cross_validation(x_train, train_labels, gnb_restricted, test_fraction = 0.2, f1_score_average = 'binary')
        print("Gaussian Naive Bayes on test data set: f1 score for each roll: ", nv_f1, "\nAverage f1 score: ", np.average(nv_f1))


    ## PART 4: Try out infrastructure for different scikit datasets
    if try_new_data:
        ## IRIS DATA SET ##

        # First load the iris dataset consisting og 150 4D samples divided in 3 classes
        iris_obj = datasets.load_iris()
        #print(iris_obj['DESCR'])
        iris_data = iris_obj['data']
        iris_labels = iris_obj['target']
        # Perform knn cross validation on the traning data
        # restrict the knn function to make it suitable for cross_validation

        leaf_size = [1, 3, 7, 10, 25, 50, 100]
      
        for leaf in [10]: # leaf_size:
            for k_nearest in [3]: #[3,5,7,9]:
                knn_restricted = lambda train_data, train_labels, test_data: k_nearest_neighbors(train_data, train_labels, test_data, \
            leaf_size = leaf, nearest_neighbors = k_nearest, runtime_information = False, binary_labels = False, labeling_warning = True)
        f1_scores = cross_validation (iris_data, iris_labels, knn_restricted, test_fraction = 0.2, f1_score_average = 'weighted')
        print('\nFor the iris dataset (150 points of dim = 4, 3 classes): \n')
        print(f'Weighted f1-scores using knn for k-nearest = {k_nearest} and leaf size = {leaf} : ', \
             f1_scores, " \nAverage f1 score: ", np.round(np.average(f1_scores),6))
        
        # Perform Gaussian Naive Bayes on the training data
        gnb_restricted = lambda data, data_labels, test: gaussian_naive_bayes(data, data_labels, test)[0]

        nb_f1 = cross_validation(iris_data, iris_labels, gnb_restricted, test_fraction = 0.2, f1_score_average = 'weighted')
        print(f'Weighted f1-scores using Gaussian Naive Bayes for k-nearest = {k_nearest} and leaf size = {leaf} : ', \
            nb_f1, " \nAverage f1 score: ", np.round(np.average(nb_f1),6))

        ## WINE DATASET ##

        # Load the wine dataset containing 178 13D points in 3 classes
        wine = datasets.load_wine()
      #  print("\n \n \n",wine['DESCR'])

        wine_data = wine['data']
        wine_labes = wine['target']

        for leaf in [3]: #  [leaf_size]:
            for k_nearest in [3]: # [3,5,7,9]:
                knn_restricted = lambda train_data, train_labels, test_data: k_nearest_neighbors(train_data, train_labels, test_data, \
            leaf_size = leaf, nearest_neighbors = k_nearest, runtime_information = False, binary_labels = False, labeling_warning = False)
                f1_scores = cross_validation (wine_data, wine_labes, knn_restricted, test_fraction = 0.2, f1_score_average = 'weighted')
                print('\nFor the wine dataset (178 points of dim = 13, 3 classes): \n')
                print(f'Weighted f1-scores using knn for k-nearest = {k_nearest} and leaf size = {leaf} : ', f1_scores, " \nAverage f1 score: ", np.round(np.average(f1_scores),6))        

        # Perform Gaussian Naive Bayes

        nb_f1 = cross_validation(iris_data, iris_labels, gnb_restricted, test_fraction = 0.2, f1_score_average = 'weighted')
        print(f'Weighted f1-scores using Gaussian Naive Bayes for k-nearest = {k_nearest} and leaf size = {leaf} : ', \
            nb_f1, " \nAverage f1 score: ", np.round(np.average(nb_f1),6))


if __name__ == "__main__":
    main()


