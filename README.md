# DisasterResponsePipelines

This an open source project to make an ETL pipeline and a Machine Learning pipeline of a model to categorize disasters messages.
Furthermore it will be implemented a browser interface to use this model.

The preparation of the data can be found in the notebooks in the preparation folder.
The cleaning process is explained throughout the notebooks but they were mainly NLP processes, such as:
1- Normalize, tokenize, removing stop words and lemmatize the messages.
2- CountVectorizer and TF-IDF transformer for the document term matrix.

The dataframes are merged into a SQL database.
The files to prepare the data and train the model can be found in the respective folders and were developed with python.
Finally the web app is design using flask, html and CSS.
