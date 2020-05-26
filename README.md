# DisasterResponsePipelines

This an open source project to make an ETL pipeline and a Machine Learning pipeline of a model to categorize disasters messages.
Furthermore it will be implemented a browser interface to use this model.

The preparation of the data can be found in the notebooks with the preparation tag.
The cleaning process is explained throughout the notebooks but they were mainly NLP processes, such as:
1- Normalize, tokenize, removing stop words and lemmatize the messages.
2- CountVectorizer and TF-IDF transformer for the document term matrix.

The dataframes are merged into a SQL database.
All code will be developed using python.
