
###############################################################################
##################### Train and Test our model #################
from sklearn import metrics
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, KFold


def cross_validate_model(clf, X_train, X_test, y_train, y_test):
    # we have extracted our features here in this function... Combine the X_train and X_test dataset
    # form a giant 1 dataset
    # cross validate that 1 giant dataset



    whole_dataset_nparray = np.concatenate((X_train.toarray(), X_test.toarray()))

    whole_output_nparray = y_train + y_test

    kfold = KFold(n_splits = 10, shuffle = True, random_state = 1)
    split_accuracies = []
    split_f_measure = []
    split_precision = []
    split_recall = []

    for train_ds_idxs, test_ds_idxs in kfold.split(whole_dataset_nparray):
        X_train_cur, X_test_cur = whole_dataset_nparray[train_ds_idxs], whole_dataset_nparray[test_ds_idxs]
        y_train = [whole_output_nparray[train_idx] for train_idx in train_ds_idxs]
        y_test = [whole_output_nparray[test_idx] for test_idx in test_ds_idxs]



        accuracy, f_measure, precision, recall = train_and_test_model(clf, X_train_cur, y_train, X_test_cur, y_test)

        split_accuracies.append(accuracy)
        split_f_measure.append(f_measure)
        split_precision.append(precision)
        split_recall.append(recall)


    ### get avg accuracy
    def get_avg(vals):
        return (sum(vals) / len(vals))

    average_accuracy = get_avg(split_accuracies)
    average_f_measure = get_avg(split_f_measure)
    average_precision = get_avg(split_precision)
    average_recall = get_avg(split_recall)

    return average_accuracy, average_f_measure, average_precision, average_recall



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
    # confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)

    # plt.imshow(confusion_matrix)

    # plt.set_cmap('jet')

    # plt.show()


    ###################### print the accuracy of our classifier ###########################
    accuracy = metrics.accuracy_score(y_test, y_predicted, normalize = True)
    # print(f'Accuracy of our classifier is : {accuracy}')

    ################# f-measure score #########################
    f_measure = metrics.f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted))
    # print(f'F-Major of our classifier is : {f_measure}')

    ################# precesion ##############################
    precision = metrics.precision_score(y_test, y_predicted, average='weighted',labels=np.unique(y_predicted))
    # print(f'Precision of our classifier is {precision}')
    recall=metrics.recall_score(y_test, y_predicted, average='weighted')
    # print(f'Recall of our classifier is {recall}')

    return accuracy, f_measure, precision, recall
   