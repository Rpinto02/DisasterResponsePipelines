import json
import random

import plotly
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Messages.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/train_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():



    # extract data needed for visuals
    rowSums = df.iloc[:, 2:].sum(axis=1)
    multiLabel_counts = rowSums.value_counts()
    multiLabel_counts = multiLabel_counts.iloc[1:]
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    categories = list(df.columns[3:].values)
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
              'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'salmon', 'darksalmon',
              'lightcoral', 'indianred', 'crimson', 'firebrick', 'coral', 'tomato','orangered', 'gold', 'orange',
              'lawngreen', 'chartreuse', 'limegreen', 'lime', 'forestgreen', 'green', 'darkgreen', 'greenyellow',
              'yellowgreen', 'springgreen']



    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=multiLabel_counts.index,
                    y=multiLabel_counts.values,
                    marker=dict(color=random.sample(colors, len(colors))),

                )
            ],

            'layout': {
                'title': 'Comments having multiple labels',

                'yaxis': {
                    'title': "Number of comments"
                },
                'xaxis': {
                    'title': "Number of labels"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=df.iloc[:, 3:].sum().values,
                    marker=dict(color=random.sample(colors, len(colors))),
                )
            ],

            'layout': {
                'title': 'Comments in each category',

                'yaxis': {
                    'title': "Number of comments"
                },
                'xaxis': {
                    'title': "Comment Type",
                    'tickangle': 50,
                    'automargin': True,
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color=['blueviolet', 'brown', 'blanchedalmond'])
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()