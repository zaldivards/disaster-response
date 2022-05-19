# Disaster Response Pipeline Project

## Project motivation

This project is intended to analyze disaster data to build a model for an API that classifies disaster messages.

## Files

- data/
    - disaster_categories.csv: the raw data containing the labels for each message.
    - disaster_messages.csv: the raw data with the messages and its genres.
## Instructions to reproduce it locally:

1. First of all, you must assure to install the dependencies:
```shell
    pip install -r requirements.txt
```

2. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database:
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

3. Run the web app: `python run.py`

## Other info
The messages data is **skewed**, hence the accuracy of the predictions will be affected. You can check the bar plot to make sure of it.

# Project preview
![Not found](.preview/ss1.png)

![Not found](.preview/ss2.png)


