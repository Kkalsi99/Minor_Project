############################################################################################
##################        LOAD DATASET      ##############################################

import os
from xml.etree import ElementTree


def load_data(xmls_directory, truth_path, txts_destination_directory):
    """ 
    Loads Pan data

    @return: 
    1. merged tweets of authors
    2. truths(read genders of the authors)
    3. author ids
    4. original tweet lengths of authors

    @TODO:
        return author_ages from this function as well
    """

    # read tweets of the authors
    # read the filenames from the xmls_dir
    xmls_filenames = sorted(os.listdir(xmls_directory))

    # xml filename = (author_id.xml)
    # to ge author id: split(filename)[0]
    author_ids = get_author_ids(xmls_filenames)

    # get the known values of gender and age from the file specified by "Truth Path"
    gender_truths, age_truths = get_truths(truth_path, author_ids)
            
    merged_tweets, original_tweet_lengths = get_author_merged_tweets_and_tweet_lengths(xmls_directory, xmls_filenames)
    

    return merged_tweets, gender_truths, age_truths, author_ids, original_tweet_lengths


def get_author_merged_tweets_and_tweet_lengths(xmls_directory, xmls_filenames):
    original_tweet_lengths = []

    merged_tweets = []

    # get filenames and read tweets
    for author_idx, xml_filename in enumerate(xmls_filenames):
        # read filename
        # construct tree of xml
        
        
        tree = ElementTree.parse(os.path.join(xmls_directory, xml_filename))
      
        root = tree.getroot()
        
        original_tweet_lengths.append([])

        tweets_of_this_author = []

        # root[0] --> first level of the tree
        # since the tweets are in 1st level 
        
        for child in root:

            tweet = child.text
            
	
            original_tweet_lengths[author_idx].append(len(tweet))

            # replace \n with lineFeed
            tweet.replace('\n', '<LineFeed>')

            tweets_of_this_author.append(tweet)

        
        # store tweets as string
        # string separated by <EndOfTweet> 
        merged_tweets_of_this_author = "<EndOfTweet>".join(tweets_of_this_author)+"<EndOfTweet>"

        merged_tweets.append(merged_tweets_of_this_author)


    return merged_tweets, original_tweet_lengths




def get_truths(truth_path, author_ids):
    gender = {}
    age = {}
    with open(truth_path, 'r') as truth_file:
        for line in truth_file:
            line.rstrip('\n')
            entry = line.split(':::')

            author_id, author_gender, author_age = entry[0:3]
            # @TODO -> make sure the age in our dataset as wellauthor_id, author_gender, author_age = entry[0], entry[1], entry[2]
            gender[author_id] = author_gender
            age[author_id] = author_age
    # we get the truths gender_truths = []
    age_truths = []
    gender_truths = []

    for author_id in author_ids:
        gender_truths.append(gender[author_id])
        age_truths.append(age[author_id])
    return gender_truths,age_truths



def get_author_ids(xmls_filenames):
    author_ids = []
    for xml_filename in xmls_filenames:
        author_ids.append(xml_filename[:-4])
    return author_ids