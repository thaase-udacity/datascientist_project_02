# Business Understanding
# This ML script use the in the ETL script created database ad creates a ML model on it
# Afterwards it stores the data into a pickl file
# It has 2 parameters which are the db name and the name of the pickl file

# import libraries
import sys
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from datetime import datetime

import nltk
nltk.download('stopwords')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# function loads date from the database and splits data in test and train data
def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - a string with the database filepath
    
    OUTPUT:
    two dataset X and y with categorical and numerical data
    '''
    print('{} connect to DB'.format(datetime.now()))
    conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    
    print('{} run SQL and store in pandas dataset'.format(datetime.now()))
    df = pd.read_sql('select * from desaster_response', con=conn)
    
    # debug info
    #print(df.info(verbose=True))
    #print(df.head(10))
    
    # split dataset in text and numerical values
    X = df.message
    y = df.select_dtypes(include=int)
    
    # debug info
    # check if X and y are same datatype
    print('X is {}'.format(type(X)))
    print('y is {}'.format(type(y)))
    
    # drop column "id" from y (selecting all INT column and deleting one is quicker then typing all column names)
    y.drop(['id'], axis=1, inplace=True)
    
    # close db connection
    conn.close()
    return X, y

# function tokenizes a text
def tokenize(text):
    '''
    INPUT:
    text - a string with a message
    
    OUTPUT:
    a list of tokens in lowercase with removed stopwords
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # lemmatize without removing stop word
    #tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens  

# creates a model with random forest
def build_model():
    '''
    OUTPUT:
    a grid search object with a pipeline and parameters using the random forest classifier
    '''
    print('{} create pipeline random forest'.format(datetime.now()))
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clm', MultiOutputClassifier(RandomForestClassifier()))
    ])    
    
    # create parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
        ,'tfidf__use_idf': [True, False]
    }
    
    #create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

# create a second model with different Classifier: AdaBoost
def build_model_2():
    '''
    OUTPUT:
    a grid search object with a pipeline and parameters using the ada boost classifier
    '''
    print('{} create pipeline AdaBoost'.format(datetime.now()))
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clm', MultiOutputClassifier(AdaBoostClassifier()))
    ])    
    
    # create parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
        ,'tfidf__use_idf': [True, False]
    }
    
    #create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

# create a third model with different Classifier: ExtraTreesClassifier
def build_model_3():
    '''
    OUTPUT:
    a grid search object with a pipeline and parameters using the extra trees classifier
    '''
    print('{} create pipeline ExtraTreesClassifier'.format(datetime.now()))
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clm', MultiOutputClassifier(ExtraTreesClassifier()))
    ])    
    
    # create parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
        ,'tfidf__use_idf': [True, False]
    }
    
    #create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def display_results(cv, y_test, y_pred):
    '''
    INPUT:
    cv - a grid search object
    y_test the test data set
    y_pred the pedicted dataset
    OUTPUT:
    Print details about the prediction quality.
    '''
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()
    print("Labels:", labels)
    print(classification_report(y_test.values, y_pred, labels=labels, target_names = y_test.columns.values))
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_) 

# function to save the model to pickl file
def save_model(model, model_filepath):
    '''
    INPUT:
    model - the predicted model
    model_filepath - the filepath where the function should store the model as pickl file
    '''
    print('{} save model to pickl file'.format(datetime.now()))
    # save the model to pickl file
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        # Version 1 with RandomForest Classifier
        # avg / total       0.73      0.47      0.53     20784
        #print('{} create gridsearch model random forest'.format(datetime.now()))
        #model = build_model()
    
        # Version 2 with a different classifier: AdaBoost
        # avg / total       0.72      0.59      0.63     20680
        # Model 2 has the best F1 Score --> use it!
        print('{} create gridsearch model ada boost'.format(datetime.now()))
        model = build_model_2()
    
        # Version 3 with a third classifier: ExtraTreesClassifier
        # avg / total       0.73      0.47      0.53     21291
        #print('{} create gridsearch model ExtraTreesClassifier'.format(datetime.now()))
        #model = build_model_3()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('{} predict on test data'.format(datetime.now()))    
        y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        display_results(model, Y_test, y_pred)
    
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
