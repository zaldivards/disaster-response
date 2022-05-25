import pickle

import pandas as pd
from flask import Flask, render_template, request
from sqlalchemy import create_engine
from waitress import serve

from graph.utils import cache_handler, get_serialized_graphs
# used by the pipeline when it's loaded.
from models.train_classifier import VerbCounterEstimator, tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
with open("./models/classifier.pkl", 'br') as f:
    model = pickle.load(f)

cache = cache_handler(df)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # create visuals
    ids, graphJSON = get_serialized_graphs(*cache())

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
    serve(app, host='0.0.0.0', port=80)


if __name__ == '__main__':
    main()
