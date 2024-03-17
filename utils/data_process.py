# dependencies import
import pandas as pd 

data = pd.read_csv("data/Online_Retail_Data_Set.csv", encoding = "unicode_escape")
print(data)

# duplicate data removal 
print("Sum of duplicate values:", data.duplicated().sum())
data = data.drop_duplicates()
print("Sum after dropping duplicate values:", data.duplicated().sum())

# handling missing data
print("Missing values:", data.isna().sum())
copy = data.copy()
print("List:", copy)

data['Description'] = data['Description'].fillna("Unknown")
data['CustomerID'] = data['CustomerID'].fillna(0)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format="%d-%m-%Y %H:%M")
data['Quantity'] = pd.to_numeric(data['Quantity'])
data['UnitPrice'] = pd.to_numeric(data['UnitPrice'])

# creating a separate column for time
data['Time'] = data['InvoiceDate'].dt.time
# creating a separate column for month
data['Month'] = data['InvoiceDate'].dt.month_name()
# creating a separate column for day name
data['Day'] = data['InvoiceDate'].dt.day_name()
# creating a column for year
data['Year']= data['InvoiceDate'].dt.year

# creating a column for total
data['Total'] = data['Quantity']*data['UnitPrice']
print("Preprocessed data:", data)