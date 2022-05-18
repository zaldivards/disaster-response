import json
from typing import List, Tuple

import pandas as pd
import plotly
from plotly.graph_objs import Bar


def _build_stacked_bar(dataset: pd.DataFrame):
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
                          dataset: pd.DataFrame) -> Tuple[List[str], str]:

    bars = _build_stacked_bar(dataset)
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
        },
        {
            'data': bars,
            'layout': {
                'height': 800,
                'title': "Distribution of Messages' categories per genre",
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
