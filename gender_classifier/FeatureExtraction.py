import preprocessor as tp
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN



def lemmatize_tweet(tweet_tokens):
    lemmatizer = WordNetLemmatizer()
    

    lemmatized_tokens = []
    for token in tweet_tokens:
        pos_token = pos_tag(word_tokenize(token))[0][1]

        wordnet_pos_tag = get_wordnet_pos(pos_token)

        lemmatized_token = lemmatizer.lemmatize(token, wordnet_pos_tag)

        lemmatized_tokens.append(lemmatized_token)


    return ' '.join(lemmatized_tokens)

def preprocess_tweet(tweet):
    """
    This function will preprocess the input tweet

    Steps for preprocessing:
        1. Lowercase the letters
        2. Replace the characters with frequency greater than 3 with 3 in a word
        3. Replace a url with Tag: <URLURL>
        4. Replace a tag mention: <UsernameMention>

    
    @TODO:
        1. Look for better preprocessing methods on the web
        2. Apply here
    """
    clean_tweet = tp.clean(tweet)
    
    # perform lemmatization
    tokenizer = TweetTokenizer()
    tweet_tokens = tokenizer.tokenize(clean_tweet)
    
    lemmatized_tweet = lemmatize_tweet(tweet_tokens)
    
    
    # remove stopwords
    preprocessed_tweet = remove_stopwords(lemmatized_tweet)
    return preprocessed_tweet


###########################################################################################################
############################# Extract features of the data provided #########################################
def extract_features(docs_train, docs_test, perform_dimensionality_reduction):
    """ 
    We will extract features from the dataset, preprocess it and return the X_train and X_test
    
    @return:
        1. X_train: Feature matrix for training data
        2. X_test: Feature matrix for test data


    @Regions of improvement:
        1. Get more features and use them to get more accurate predictions 
   
    """
    word_ngram_range = (1, 4)
    char_ngram_range = (2, 5)

    '''
    Build an n grams vectorizer with word_n_gram_range and char_n_gram_range
    '''

    ngrams_vectorizer = create_n_grams_vectorizer(word_ngram_range, char_ngram_range)


    # use the n_gram vectorizer to form the train and test dataset
    X_train = ngrams_vectorizer.fit_transform(docs_train) #it will take a lot of time... i think
    X_test = ngrams_vectorizer.transform(docs_test)
    print("Performed fitting of data")
    ############ dimensionality reduction ################
    
    if(perform_dimensionality_reduction == True):                                 
        X_train, X_test = perform_dimensionality_reduction(X_train, X_test)
       


    # print(docs_train[0])
    return X_train, X_test




def create_n_grams_vectorizer(word_ngram_range, char_ngram_range):
    word_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                    analyzer='word',
                                    ngram_range=word_ngram_range,
                                    min_df=2,
                                    use_idf=True, 
                                    sublinear_tf=True)
    print(f'Created a word vectorizer')
    char_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                     analyzer='char', 
                                     ngram_range=char_ngram_range,
                                     min_df=2, 
                                     use_idf=True, 
                                     sublinear_tf=True)
    print(f'Created a char vectorizer')




    ###############################################################################################
    ################## Count vectorizer -> which just computes the count of tokens ################


    '''
    Merge the two vectorizers using a pipeline
    '''
    return Pipeline([('feats', FeatureUnion([
                                                        ('word_ngram', word_vectorizer),
                                                         ('char_ngram', char_vectorizer)
                                                         ])),
                                 # ('clff', LinearSVC(random_state=42))
                                 ])




def perform_dimensionality_reduction(X_train, X_test):
    
    print("Performing dimensionality reduction")
    # use TruncatedSVD to reduce dimensionality of our dataset
    svd = TruncatedSVD(n_components = 300, random_state = 42)

    X_train = svd.fit_transform(X_train)
    
    X_test = svd.transform(X_test)
    
    print("Performed dimensionality reduction")