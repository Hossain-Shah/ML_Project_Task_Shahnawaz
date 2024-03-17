from flask import Flask, request, jsonify
import sqlite3
import spacy

app = Flask(__name__)

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define the SQLite database filename
db_filename = 'data/online_retail.db'

# Function to process the user message using NLP
def process_message(message):
    # Process the message using spaCy
    doc = nlp(message)
    
    # Extract entities
    entities = [ent.text for ent in doc.ents]
    
    # Identify intent based on keywords or patterns
    intent = "product_inquiry" if "product" in message else "general_inquiry"
    
    return intent, entities

# Function to get product recommendations based on customer history
def get_product_recommendations(customer_id):
    # Connect to the database
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Query the database for customer purchase history
    cursor.execute("SELECT Description FROM OnlineRetail WHERE CustomerID = ?", (customer_id,))
    purchase_history = cursor.fetchall()

    recommendations = [row[0] for row in purchase_history]
    
    conn.close()
    
    return recommendations

# Route to handle incoming messages
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message'].lower()
    action = request.json.get('action')
    
    # Process user message using NLP
    intent, entities = process_message(user_message)
    
    # Connect to the database
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Query the database based on user intent
    if intent == 'product_inquiry':
        # Perform product-related query
        cursor.execute("SELECT * FROM OnlineRetail WHERE LOWER(Description) LIKE ?", ('%'.join(entities),))
    else:
        # Perform general inquiry query
        cursor.execute("SELECT * FROM OnlineRetail WHERE LOWER(CustomerID) = ?", (user_message,))
    
    results = cursor.fetchall()
    
    # Get product recommendations if the action is for recommendations
    recommendations = []
    if action == 'recommendations':
        recommendations = get_product_recommendations(user_message)
    
    conn.close()
    
    # Format the response message
    if results:
        response = {
            'status': 'success',
            'message': f"Customer details found: {results}",
            'recommendations': recommendations
        }
    else:
        response = {
            'status': 'error',
            'message': 'No matching information found.'
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
