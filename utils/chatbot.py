from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

# Define the SQLite database filename
db_filename = 'data/online_retail.db'

# Route to handle incoming messages
@app.route('/chat', methods=['POST'])
def chat():
    # Get user message from the request
    user_message = request.json['message'].lower()
    
    # Connect to the database
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Query the database based on user message
    cursor.execute("SELECT * FROM OnlineRetail WHERE LOWER(CustomerID) = ?", (user_message,))
    results = cursor.fetchall()
    
    conn.close()
    
    # Format the response message
    if results:
        response = {
            'status': 'success',
            'message': f"Customer details found: {results}"
        }
    else:
        response = {
            'status': 'error',
            'message': 'Customer not found.'
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
