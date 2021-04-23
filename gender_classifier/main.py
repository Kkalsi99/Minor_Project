#!/usr/bin/env python3
# sklearn libraries
from TrainAndTest import cross_validate_model
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
# LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
# neural network classifier
from sklearn.neural_network import MLPClassifier


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
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        random_state=1)
    cross_validate_model(clf, X_train, X_test, y_train, y_test)

    print("################LR##############")
    clf = LogisticRegression(random_state=0)
    cross_validate_model(clf, X_train, X_test, y_train, y_test)
    print("###############SVM###############")
    clf = LinearSVC(random_state=42, tol=0.3)
    cross_validate_model(clf, X_train, X_test, y_train, y_test)
    print("###############DT###############")
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    cross_validate_model(clf, X_train, X_test, y_train, y_test)
    print("###############RF###############")
    clf = RandomForestClassifier(n_estimators=1000)
    cross_validate_model(clf, X_train, X_test, y_train, y_test)


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
