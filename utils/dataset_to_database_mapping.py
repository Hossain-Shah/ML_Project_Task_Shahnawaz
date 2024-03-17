import sqlite3
import csv

# Define the SQLite database filename
db_filename = 'data/online_retail.db'

# Define the database schema and create the table
def create_database():
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Create OnlineRetail table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS OnlineRetail (
            InvoiceNo TEXT,
            StockCode TEXT,
            Description TEXT,
            Quantity INTEGER,
            InvoiceDate TEXT,
            UnitPrice REAL,
            CustomerID TEXT,
            Country TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Load data from CSV file and insert into the database
def load_data(filename):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    with open(filename, 'r', encoding='latin-1') as file:  
        csv_reader = csv.reader(file)
        next(csv_reader)  
        for row in csv_reader:
            cursor.execute('''
                INSERT INTO OnlineRetail (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)
    
    conn.commit()
    conn.close()

# Create the database and load data from the CSV file
def main():
    create_database()
    load_data('data/Online_Retail_Data_Set.csv')
    print("Database created and data loaded successfully.")

if __name__ == "__main__":
    main()
