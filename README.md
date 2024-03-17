# Product retail automated bot

## Environment setup:
* python -m venv env
* source env/bin/activate
* pip install -r requirements.txt

## Data process:
* python utils/data_process.py

## Exploratory data analysis:
* python utils/data_analysis.py

## Model build:
* python utils/model_development.py

## Model evaluation:
* python utils/evaluation.py

## Database mapping:
* python utils/dataset_to_database_mapping.py

## Plain chatbot development:
* python utils/chatbot.py

## NLP Featured recommendation oriented chatbot development:
* python utils/automated_chatbot.py

## Utilities:

-> "data" folder gathers dataset and converted database in sqlite format.

-> "models" folder gathers retail model trained on "Online_Retail_Data_Set.csv" dataset.

-> "outputs" folder gathers figures of data analytics of the "Online_Retail_Data_Set.csv" dataset.  

-> "utils" folder gathers all necessary source codes.

## Features:

|_ Trained "Online_Retail_model" determines customer purchase behavior based on "InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country" these terms which is basically an ensembled "Random forest classifier".

|_ Plain chatbot provides information based on user message which is just an SQL query based on CustomerID.

|_ Automated chatbot provides information based on user message and recommendations natural language processing oriented.

# N.T.B: Removed model for large size, LFS dependency 
