import sys
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import sqlite3

#messages_filepath = 'data/disaster_messages.csv'
#categories_filepath = 'data/disaster_categories.csv'

def load_data(messages_filepath, categories_filepath):
    '''loads the message and categories dataframes'''

    messages = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)


    return messages, categories_df;

def only_numbers(cell):
    '''takes an array, cell, and cleans all the non alphanumeric character,
    transforms them into a list of integers and returns that list
    '''

    cleaned_cell = re.sub(r'[a-zA-Z_-]+', '', cell)
    cleaned_cell = list(map(int,cleaned_cell.split(';')))
    return cleaned_cell

def clean_data(df,column):
    '''takes in a dataframe where a column is composed by words followed by numbers,
    and transforms it into a dataframe where the words(first row only) are the columns and the numbers are in
    the rows of the respective columns
    Also removes the words who have a zero count, that is the number which followed it was always zero.

    column - column to clean (string)
    df - dataframe to clean
    '''

    #reading the first row of the dataframe and transforming into an array of words
    column_features = re.sub(r"[-01;]", " ",df[column][0])
    column_features = word_tokenize(column_features )

    #creating a temporary dataframe to remove the words and keep the numbers and transform it into a list of lists
    temp_df = df[column]
    fill_df = []
    for i in range (0,len(temp_df)):
        fill_df.append(only_numbers(temp_df[i]))

    #creating the new dataframe with the expected result
    new_df = pd.DataFrame(np.array(fill_df), columns=[column_features])

    #removing the features with only zeros
    new_df = new_df.loc[:,new_df.sum(axis=0)>0]

    #removing the rows with all zero values
    new_df = new_df[new_df.sum(axis=1) > 0]

    #removing the rows where the features are different than 0 or 1
    new_df = new_df[(new_df.iloc[:,0] == 0) | (new_df.iloc[:,0] == 1)]

    return new_df;

def preparation(messages,cleaned_categories):
    '''takes care of the merging process and final clean up of the dataframes
    '''
    #merging the two dataframes
    df = messages.merge(cleaned_categories,left_index=True, right_index=True)

    #fixing column names
    df.rename(columns=lambda col: ''.join(col), inplace=True)

    #dropping the columns that will not be used
    df = df.drop(columns=['id','original'])

    #creating dummies from the genre column
    #df = pd.get_dummies(df,columns=['genre'],drop_first=True)

    # dropping duplicate rows
    df.drop_duplicates(keep=False,inplace=True)

    #dropping the related feature - see Data_Exploration_Analysis for further details on why
    df = df.drop(columns=['related'])

    return df;

def save_data(df, database_filename):
    '''exports the cleaned dataframe to a SQL database'''

    database_name = 'Messages'
    connection = database_name+'.db'
    #opening connection and cursor
    conn = sqlite3.connect(connection)
    c = conn.cursor()
    conn.commit()

    #transforming the dataframe to a sql database
    df.to_sql(database_name, conn, if_exists="replace")

    #closing connections
    c.close()
    conn.close()



def main():

    if len(sys.argv) == 4:
        messages_filepath , categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        column = 'categories'
        cleaned_categories = clean_data(categories_df, column)
        df = preparation(messages, cleaned_categories)
        
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

#messages_filepath = 'c:/udacity/DisastersResponsePipeline/data/disaster_messages.csv'
#categories_filepath = 'c:/udacity/DisastersResponsePipeline/data/disaster_categories.csv'
#database_filepath = 'Messages.db'

if __name__ == '__main__':
    main()