# datascientist_project_02
2024.03.19 Torsten Haase

Udacity Datascience course - Delivery 02 desaster response pipeline

Project uses the following libaries:

pandas
numpy
matplotlib

Motivation of the project is to create a whole ML pipeline together with a web app.

The following files are included:

readme.md -> this files

data folder:
data/categories.csv -> which holds the categories for each post
data/messages.csv -> which holds all message data
data/desasterresponse.db -> in this sqlite db all data is stored

model folder:



Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

Thanks to the udacity team for the data science learning program. 
