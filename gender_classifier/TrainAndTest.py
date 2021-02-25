
###############################################################################
##################### Train and Test our model #################
from sklearn import metrics
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt


def train_and_test_model(clf, X_train, y_train, X_test, y_test):
    # training phase
    # fit the X_train, y_train on the clf
    clf.fit(X_train, y_train)

    '''
    Predict the output of the test set
    '''
    y_predicted = clf.predict(X_test)

    '''
    Build the confusion matrix
    '''
    confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)

    plt.imshow(confusion_matrix)

    plt.set_cmap('jet')

    plt.show()


    ###################### print the accuracy of our classifier ###########################
    accuracy = metrics.accuracy_score(y_test, y_predicted, normalize = True)
    print(f'Accuracy of our classifier is : {accuracy}')

    ################# f-measure score #########################
    f_measure = metrics.f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted))
    print(f'F-Major of our classifier is : {f_measure}')

    ################# precesion ##############################
    precision = metrics.precision_score(y_test, y_predicted, average='weighted',labels=np.unique(y_predicted))
    print(f'Precision of our classifier is {precision}')
    recall=metrics.recall_score(y_test, y_predicted, average='weighted')
    print(f'Recall of our classifier is {recall}')
   