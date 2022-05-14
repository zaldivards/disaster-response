import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str,
              categories_filepath: str) -> pd.DataFrame:
    """Loads the datasets from the filesystem

    Args:
        messages_filepath (str): path of the messages dataset
        categories_filepath (str): path of the categories dataset

    Returns:
        pd.DataFrame: the merged dataset using the messages and categories
        dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df: pd.DataFrame):
    """Extract categories in separate columns, merge them with
    the main dataset and than those are concatenated

    Args:
        df (pd.DataFrame): the merged messages and categories(in one column)

    Returns:
        pd.DataFrame: the main messages with the new separated
        categories(one per column)
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda cat: cat[:-2])
    categories.columns = category_colnames
    categories = categories.apply(
        lambda series: series.str.extract(r'(\d)+')[0].astype('int8'))
    df = df.drop('categories', axis=1)
    return pd.concat([df, categories], axis=1).drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str):
    """Save the data in an sqlite db

    Args:
        df (pd.DataFrame): the cleaned dataset
        database_filename (str): the database name
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath \
            = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
