# System tools
import os
import sys
sys.path.append(os.path.join("utils"))

# Data tools
import pandas as pd
import numpy as np
import utils.classifier_utils as clf

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import re

# Visualisation  tools
import matplotlib.pyplot as plt
import seaborn as sns

# Surpress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

 
    

def fake_or_real():

    # Get data
    filename = os.path.join("input", "fake_job_postings.csv")
    data = pd.read_csv(filename)
    # Now this time I am balancing the data
    blanced_data = clf.balance(data, 800) # Taking 800 data points
    # Making the dataset more efficient to work with (as well as more manageable) 
    # Replacing NA cells with space
    data.fillna(" ",inplace = True)
    # Gathering the most useful information in a new column called "info_text"
    data['info_text']=data['title']+" "+data['location']+" "+data['department']+" "+data['company_profile']+" "+data['description']+" "+data['requirements']+" "+data['benefits']+" "+data['employment_type']+" " +data['required_education']+" "+data['industry']+" "+data['function'] 

    # keep only the columsn we want    
    data = data[["job_id", "fraudulent", "info_text"]]

    # Performing regex and clearing the info_text, so that it can be processed more accurate 
    data['info_text']= data['info_text'].str.replace('\n',' ') #
    data['info_text']= data['info_text'].str.replace('\r',' ') #
    data['info_text']= data['info_text'].str.replace('\t',' ') #
    data['info_text'] = data['info_text'].apply(lambda x: re.sub(r'[0-9]',' ',x))
    data['info_text'] = data['info_text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
    data['info_text']= data['info_text'].apply(lambda s:s.lower() if type(s) == str else s)
    data['info_text']= data['info_text'].str.replace('  ',' ')

    # At last, renaming the 0 and 1 with real and fake
    data["fraudulent"].replace({0:"real job", 1:"fake job"}, inplace = True)

    data.head()

    # Creating new variables where I take the data out of the dataframe in order to work with them 
    X = data["info_text"]
    y = data["fraudulent"]

    # Making a train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,                  # Text for the model
                                                        y,                  # Classification labels
                                                        test_size = 0.2,    # Defining an 80/20 split 
                                                        random_state = 42)  # Defining random state for reproducability 



    # Choosing Tfid vectorization to transform the text to feature vectors, so that it can be used as an input to estimator
    # Why Tfid? Because it also provides the importance of words in the "documents", rather than just the frequency

    vectorizer = TfidfVectorizer(ngram_range = (1,2),  # 1 = individual words only, 2 = either individual words or bigrams
                                 max_df = 0.95,        # df = document frequency. 0.95 = get rid of all words that occur in over 95% of the document
                                 min_df = 0.05,        # 0.05 = get rid of all words that occur in lower than 0.5% of the document
                                 max_features = 500)   # keep only top 500 features (words)



    # Fitting the data
    # First we fit to the training data
    X_train_feats = vectorizer.fit_transform(X_train)

    # Then transforming the test data 
    X_test_feats = vectorizer.transform(X_test) # I don't want to fit the test data, because it has to be tested on the training data 

    # Get feature names (words from the dataset)
    feature_names = vectorizer.get_feature_names()

    # Using the MLPClassifier ... 
    classifier = MLPClassifier(random_state =42, 
                        hidden_layer_sizes = (32,), # The number of nodes I wish to have in the Neural Network Classifier.
                        activation = "logistic", # Using the logistic activation function
                        max_iter = 500) # The maximum number of iterations 

    # Fitting training data to the network 
    classifier.fit(X_train_feats, y_train)

    # Getting the classifiers predictions from the test data
    y_pred = classifier.predict(X_test_feats)


    # Getting the classification report 
    classification_report = metrics.classification_report(y_test, y_pred)
    print(classification_report)

    with open('output/MLPClassifier.txt', 'w') as my_txt_file:
        my_txt_file.write(classification_report)
        
        
    
    # Further evaluation 

    # For further evaluation i will use Scikit-learns different evaluation tools from logistic regression.
    # Cross validation --> in order to test a number of different train-test splits and finding the average scores.
    # Show_features --> in terms of inspecting what features are most informative when trying to predict a label.
    
    X_vect = vectorizer.fit_transform(X)
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0)
    # n_splits = shuffled and split the data 100 times 

    # Usinf logistic regression as estimator 
    estimator = LogisticRegression(random_state = 42)
    cross_val = clf.plot_learning_curve(estimator, title, X_vect, y, cv = cv, n_jobs = 4)    
    
    # The final result saved as a picture
    plt.savefig('output/Cross_validation.png')

    # Results: 
    # Overfitting (big gap) = high variance 
    # Underfitting (small gap) = high bias 
    # The second plot shows the times required by the models to train with various sizes of training dataset. 
    # The third plot show how much time was required to train the models for each training sizes.
    
    # Making a new classifier with logistic regression in order to use show_features
    lr_classifier = LogisticRegression(random_state = 42).fit(X_train_feats, y_train)
    # what was the classifier learned from the model  from the labels 
    inspect_features = clf.show_features(vectorizer, y_train, lr_classifier, n = 20) 

    # inspect_features will be printed to commandline 
    # Results: These are the features (words) used to predict wether the given job post i real or fake


    print("Script succeeded, results can be seen in the output-folder")

fake_or_real()

