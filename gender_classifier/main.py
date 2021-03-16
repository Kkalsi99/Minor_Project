#!/usr/bin/env python3
# sklearn libraries
from TrainAndTest import train_and_test_model
from FeatureExtraction import preprocess_tweet, extract_features
from DataInput import load_data
from sklearn import utils
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import cross
from numpy.lib.function_base import average
from numpy.lib.utils import who
from sklearn.model_selection import train_test_split, KFold

# skearn classifiers
from sklearn.neighbors import KNeighborsClassifier
# decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# random forest classifier
from sklearn.ensemble import RandomForestClassifier

# naive bayes classifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB, BernoulliNB

# support vector classifier
from sklearn.svm import SVC, LinearSVC


# nltk libraries


# others
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# from local modules


### global configurations #######
config2018 = {
    'dataset_name': 'PAN 2018 English',
    'xmls_directory': '../data/pan18-author-profiling-training-dataset-2018-02-27/en/text/',
    'truth_path': '../data/pan18-author-profiling-training-dataset-2018-02-27/en/en.txt',
    'txts_destination_directory': '../data/pan18-author-profiling-training-dataset-2018-02-27/en',
}
config2015 = {
    'dataset_name': 'PAN 2015 English',
    'xmls_directory': '../data/pan15-author-profiling-training-dataset-english-2015-04-23/en/text/',
    'truth_path': '../data/pan15-author-profiling-training-dataset-english-2015-04-23/en/en.txt',
    'txts_destination_directory': '../data/pan15-author-profiling-training-dataset-english-2015-04-23/en',
}


############################################################################################
####################3 Preprocess each tweet ###############################################


###################################################################
def generate_output(merged_tweets, truths, author_ids, original_tweet_lengths):
    docs_train, docs_test, y_train, y_test, author_ids_train, author_ids_test, original_tweet_lengths_train, original_tweet_lengths_test\
        = train_test_split(merged_tweets, truths, author_ids, original_tweet_lengths, test_size=0.2, random_state=2, stratify=truths)
    print("Performed train test split")

    # maintain the order in dataset
    author_ids_train, docs_train, y_train, original_tweet_lengths_train = [list(tuple) for tuple in zip(*sorted(zip(
        author_ids_train, docs_train, y_train, original_tweet_lengths_train)))]
    # Sort the test set
    author_ids_test, docs_test, y_test, original_tweet_lengths_test = [list(tuple) for tuple in zip(*sorted(zip(
        author_ids_test, docs_test, y_test, original_tweet_lengths_test)))]

    # extract features from the dataset
    X_train, X_test = extract_features(
        docs_train, docs_test, perform_dimensionality_reduction=False)
    print("Successfully extracted features from the documents")

    # we have extracted our features here in this function... Combine the X_train and X_test dataset
    # form a giant 1 dataset
    # cross validate that 1 giant dataset

    whole_dataset_nparray = np.concatenate(
        (X_train.toarray(), X_test.toarray()))

    whole_output_nparray = y_train + y_test

    print("###############SVM###############")
    clf = LinearSVC(random_state=42, tol=0.3)
    Kfolding(clf, X_train, X_test, y_train, y_test)
    print("###############DT###############")
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    Kfolding(clf, X_train, X_test, y_train, y_test)
    print("###############RF###############")
    clf = RandomForestClassifier(n_estimators=1000)
    Kfolding(clf, X_train, X_test, y_train, y_test)


def Kfolding(clf, X_train, X_test, y_train, y_test):
    whole_dataset_nparray = np.concatenate(
        (X_train.toarray(), X_test.toarray()))

    whole_output_nparray = y_train + y_test

    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    split_accuracies = []
    split_f_measure = []
    split_precision = []
    split_recall = []

    for train_ds_idxs, test_ds_idxs in kfold.split(whole_dataset_nparray):
        X_train_cur, X_test_cur = whole_dataset_nparray[train_ds_idxs], whole_dataset_nparray[test_ds_idxs]
        y_train = [whole_output_nparray[train_idx]
                   for train_idx in train_ds_idxs]
        y_test = [whole_output_nparray[test_idx] for test_idx in test_ds_idxs]

        accuracy, f_measure, precision, recall = train_and_test_model(
            clf, X_train_cur, y_train, X_test_cur, y_test)

        split_accuracies.append(accuracy)
        split_f_measure.append(f_measure)
        split_precision.append(precision)
        split_recall.append(recall)
    ######################################################################
    # building classifiers

    # get avg accuracy
    average_accuracy = sum(split_accuracies) / len(split_accuracies)
    average_f_measure = sum(split_f_measure) / len(split_f_measure)
    average_precision = sum(split_precision) / len(split_precision)
    average_recall = sum(split_recall) / len(split_recall)

    print(f'Average accuracy of our classifier : {average_accuracy}')
    print(f'Average F-Measure of our classifier : {average_f_measure}')
    print(f'Average Precision of our classifier : {average_precision}')
    print(f'Average Recall of our classifier : {average_recall}')

############### Main function ######################


def main():
    print("Starting the project...")

    # 1 -> Read the data from the files
    # Select dataset
    # 2015llR
    merged_tweets, gender_truths, age_truths, author_ids, original_tweet_lengths = load_data(
        config2015['xmls_directory'], config2015['truth_path'], config2015['txts_destination_directory'])
    # 2018
    ## merged_tweets, truths, author_ids, original_tweet_lengths = load_data(config2018['xmls_directory'], config2018['truth_path'], config2018['txts_destination_directory'])
    print("Loaded Pan data")

    print("Generating output for author Genders")
    generate_output(merged_tweets, gender_truths,
                    author_ids, original_tweet_lengths)

    print('\n\n')
    print("Generating output for Author Age")
    generate_output(merged_tweets, age_truths,
                    author_ids, original_tweet_lengths)
    # perform test train split


if __name__ == "__main__":
    main()
