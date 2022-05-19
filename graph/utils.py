import json
from typing import Callable, List, Tuple

import pandas as pd
import plotly
from plotly.graph_objs import Bar, Pie


def cache_handler(df: pd.DataFrame) -> Callable:
    """Extract and transform the data needed for the visuals

    Args:
        df (pd.DataFrame): the dataset from sqlite

    Returns:
        Callable: a new function that will return the data previously
        extracted and transformed
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cats = pd.melt(df, id_vars=['genre'],
                   value_vars=df.columns[4:],
                   var_name='cats').groupby(['genre', 'cats'],
                                            as_index=False)['value'].sum()

    def handler():
        return genre_names, genre_counts, cats
    
    return handler
    

def _build_stacked_bar(dataset: pd.DataFrame) -> List[Bar]:
    """Build several bar plots for later use to build as a stacked bar plot

    Args:
        dataset (pd.DataFrame): a dataset with the genres and categories

    Returns:
        List[Bar]: a list with the built bar plots
    """
    data = dataset.groupby('genre')
    figures = []
    for current_element in data:
        current_df: pd.DataFrame = current_element[1]
        bar = Bar(x=current_df.value, y=current_df.cats,
                  name=current_df.genre.unique()[0], orientation='h'
                  )
        figures.append(bar)
    else:
        return figures


def get_serialized_graphs(genre_names: List[str],
                          genre_counts: pd.Series,
                          cats_dataset: pd.DataFrame) -> Tuple[List[str], str]:
    """Builds the expected data structure to render the visuals on the home page

    Args:
        genre_names (List[str]): the unique names of the genres
        genre_counts (pd.Series): the number of messages per genre
        cats_dataset (pd.DataFrame): the dataset containing the number of
        messages per genre and category

    Returns:
        Tuple[List[str], str]: the ids of each visual and the serialized 
        visuals ready to be render on the home page
    """

    bars = _build_stacked_bar(cats_dataset)
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hole=0.4
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
                'annotations': [{'text': 'Genres', 'showarrow': False}]
            }
        },
        {
            'data': bars,
            'layout': {
                'height': 800,
                'title': "Distribution of message categories by genre",
                'yaxis': {
                    'title': "Categories",
                    'automargin': True
                },
                'barmode': 'stack',
                'xaxis': {
                    'title': "Categories count",
                    'range': (-100, 22_000)
                },
                'legend': {
                    'title': {'text': 'Genre', 'font': {'size': 15}}
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return ids, graphJSON
