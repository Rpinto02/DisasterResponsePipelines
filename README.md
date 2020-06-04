# Disaster Response Pipelines

## Introduction
The project is to create a web interface where you have to enter a disaster message and it will categorize the type of message using natural language processing. This will help in effective response and action to be taken to each message. This project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset is a combination of pre-labelled tweet and messages from real-life disaster.

The decision making behind the cleaning of the data as well as the exploratory analysis and model building process can all be found inside the notebooks in the preparation directory.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Getting started

### Dependencies

* Python 3.8
* Scikit-Learn 023.1
* NTLK, re
* SQLalchemy, sqlite3
* random, statistics
* Pandas, Numpy
* Json
* Joblib
* Plotly
* Flask


### Executing Program

* Run the following commands in the project's root directory to set up your database and model.
  * To run ETL Pipeline that cleans data and stores in database
  ```
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisastersResponse.db table_name
  ```
  * To run ML pipeline that trains classifier and saves.
  ```
  python model/train_classifier.py data/DisastersResponse.db table_name model/trained_model.pkl
  ```
* Run the following command in the app's directory to run your web app.
```
python run.py ../data/DisastersResponse.db table_name ../model/trained_model.pkl
```
* Go to http://localhost:3001/


## License
[MIT](https://opensource.org/licenses/MIT)

## Issue/Bug

Please open issues on github to report bugs or make feature requests.
