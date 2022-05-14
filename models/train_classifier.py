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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath) -> Tuple[np.array, np.array, Iterable]:
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * from messages', engine)
    X = df.message.values
    Y = df.loc[:, 'related':]
    return X, Y.values, Y.columns


def tokenize(text):
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


def build_model():
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=20))
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', model)
    ])
    return pipeline


def evaluate_model(model: Pipeline, X_test: np.ndarray,
                   Y_test: np.ndarray, category_names: Iterable):
    
    prediction = model.predict(X_test)
    
    report = classification_report(np.hstack(Y_test),
                                   np.hstack(prediction),
                                   labels=category_names)
    print(report)


def save_model(model: Union[MultiOutputClassifier, Pipeline],
               model_filepath: str):
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
