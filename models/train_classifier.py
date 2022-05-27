import pickle
import re
import sys
from typing import Iterable, Tuple, Union

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger',
               'stopwords', 'omw-1.4'])


class VerbCounterEstimator(BaseEstimator, TransformerMixin):

    """Custom transformer for extra feature engineering.
    Counts the number of verbs within the messages
    """

    def count_verbs(self, text) -> int:
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()).strip()
        tokens = word_tokenize(text)

        pos_tags = nltk.pos_tag(tokens)

        result = sum(part_speech[1] in ('VB', 'VBP')
                     for part_speech in pos_tags)

        return result

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.count_verbs)

        return pd.DataFrame(X_tagged)


def load_data(database_filepath: str) -> Tuple[np.array, np.array, Iterable]:
    """Loads the preprocessed dataset from sqlite

    Args:
        database_filepath (str): The path of the database

    Returns:
        Tuple[np.array, np.array, Iterable]: Tuple with the dependent and
        independent variables, and the categories' labels
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * from messages', engine)
    X = df.message.values
    Y = df[df.columns[4:]]
    columns = Y.columns
    return X, Y, columns


def tokenize(text: str) -> list:
    """Tokenizes by words, removes the stop words and applies a
    lemmatizer process to prepare input for further processing

    Args:
        text (str): Input text

    Returns:
        list: tokenized input
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()).strip()

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    lemmatized = []
    for token in tokens:
        processed_token = lemmatizer.lemmatize(token)
        processed_token = lemmatizer.lemmatize(processed_token, pos='v')
        lemmatized.append(processed_token)
    return lemmatized


def build_model() -> GridSearchCV:
    """Prepares a pipeline with components for preprocessing and modeling

    Returns:
        Pipeline: The pipeline which will be used as preprocessor and predictor
    """
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=20))

    pipeline = Pipeline([
        ('feature_transformer', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('verb_counter', VerbCounterEstimator())
        ])),

        ('classifier', model)
    ])

    parameters = {
        'classifier__estimator__n_estimators': [5, 10, 15, 20],
        'classifier__estimator__max_depth': [None, 10, 25, 50, 100],
        'feature_transformer__text_pipeline__tfidf__norm': ['l1', 'l2'],
        'feature_transformer__text_pipeline__vectorizer__max_features':
            [None, 10, 50, 100]
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=3)


def evaluate_model(model: Pipeline, X_test: np.ndarray,
                   Y_test: np.ndarray, category_names: Iterable):
    """Evaluates the model to look for bias and/or variance

    Args:
        model (Pipeline): the predictor
        X_test (np.ndarray): the independent variable
        Y_test (np.ndarray): _the dependent variable
        category_names (Iterable): the categories' labels
    """
    prediction = model.predict(X_test)

    report = classification_report(Y_test,
                                   prediction,
                                   target_names=category_names)
    print(report)


def save_model(model: Union[MultiOutputClassifier, Pipeline, GridSearchCV],
               model_filepath: str):
    """Saves the model as a pickle object

    Args:
        model (Union[MultiOutputClassifier, Pipeline]): the predictor
        model_filepath (str): name of the pickle file
    """
    with open(model_filepath, mode='bw') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Best params:', model.best_params_)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
