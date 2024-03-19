# Business Understanding
# This ETL script loads the data and does all necessary data wrangling.
# Afterwards it loads the data into an SQL DB
# It has 2 parameter which are the file path for the two files "categories" and "messages"

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# this function take the filepath of both files and return two pd objects
def load_data(messages_filepath, categories_filepath):
    print("Load messages")
    messages = pd.read_csv(messages_filepath)
    print("Load categories")
    categories = pd.read_csv(categories_filepath)
    return messages, categories

# this function cleans and transforms the data, it takes the two dataset categories
# and messages and returns one clean and joined dataset
def clean_data(messages, categories):
    #use split and create new colums from categories column
    categories_1 = categories[['categories']].squeeze().str.split(';', expand=True)
    
    # Go through the first row, split the value with "-" and use the first part as a new column name
    for i in range(categories_1.shape[1]):
        #debug info
        print("column: {} name old: {} name new: {}".format(i,categories_1.columns[i],categories_1.iloc[0,i].split('-')[0]))
        #rename
        categories_1.rename(columns={i:categories_1.iloc[0,i].split('-')[0]}, inplace = True)
    
    # take a look at the dataset
    # --> value of columns still holds the column name, column is text not numerical
    # print(categories.head(10)
    
    #remove the text and keep 0 or 1, change to numerical
    def get_value(x):
        x = int(x[-1:])
        # 2 found in first column -> 2 becomes 1
        if x==2:
            x=1
        return x

    #apply function to categories dataset
    categories_1 = categories_1.applymap(get_value)
        
    # take a look at the dataset
    # print(categories.head(10)
    # categories_1.info(verbose=True)
    # categories_1.describe()
    
    # --> column values 2 in column "related", need to be part of the function above
    # --> column "child_alone" has min=max=0 -> can be dropped
    categories_1.drop(['child_alone'], axis=1, inplace=True)

    # concatenate the categories back together
    categories = pd.concat([categories,categories_1] , axis=1)
    
    # delete "old" categories column
    categories.drop(['categories'], axis=1, inplace=True)
       
    # Merge messages and categories
    df = messages.merge(categories, on='id')
    
    # debug info
    print("duplicates: {}".format(df.duplicated().sum()))

    # delete duplicates
    df.drop_duplicates(inplace=True)

    # debug info
    print("duplicates: {}".format(df.duplicated().sum()))
    
    # debug info
    #df.info(verbose=True)
    return df

def save_data(df, database_filename):
    # store dataset in sql lite db
    # standard udacity database file name doesn't fit to use engine -> add the sqlite:///
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('desaster_response', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()