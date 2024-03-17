import csv
import random
import math
import pickle  # For model serialization
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier

# Define Node class for decision tree
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value at leaf node (for classification)

# Define Random Forest Classifier class
class RandomForestClassifier:
    def __init__(self, n_trees, max_depth, min_samples_split, n_features):
        self.n_trees = n_trees  # Number of trees in the forest
        self.max_depth = max_depth  # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.n_features = n_features  # Number of features to consider for each split
        self.trees = []  # List to hold the decision trees

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Create a decision tree and train it with bootstrap samples
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          max_features=self.n_features)
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            bootstrap_X, bootstrap_y = X[bootstrap_indices], y[bootstrap_indices]
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Use the most frequent prediction among the trees as the final prediction
        y_pred = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)
        return y_pred

    # Save the trained model to a file
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved as '{filename}'.")

    # Load a saved model from a file
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from '{filename}'.")
        return model

# Load data from CSV file
def load_csv(filename, encoding='utf-8'):
    dataset = []
    with open(filename, 'r', encoding=encoding, errors='replace') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            dataset.append(row)
    return dataset, header

# Preprocess data and encode categorical features
def preprocess_data(dataset):
    # For simplicity, we will assume that the dataset is already preprocessed and encoded
    X = np.array(dataset[:, [3, 5]], dtype=float)  # Features: Quantity, UnitPrice
    
    # Replace empty values in the 'CustomerID' column with -1
    y_values = dataset[:, 6]
    y_values[y_values == ''] = '-1'  # Replace empty strings with '-1'
    y = np.array(y_values, dtype=int)  # Convert to integer
    
    return X, y


# Main function to load data, preprocess, train and evaluate the model
def main():
    # Load data from CSV file
    filename = 'data/Online_Retail_Data_Set.csv'
    dataset, header = load_csv(filename)
    print("Dataset loaded successfully.")
    print("Header:", header)

    # Preprocess data
    X, y = preprocess_data(np.array(dataset))

    # Split data into train and test sets (80% train, 20% test)
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Define hyperparameters for Random Forest Classifier
    n_trees = 100
    max_depth = 10
    min_samples_split = 2
    n_features = int(math.sqrt(X.shape[1]))  # Number of features to consider for each split

    # Initialize and train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_trees=n_trees, max_depth=max_depth,
                                           min_samples_split=min_samples_split, n_features=n_features)
    rf_classifier.fit(X_train, y_train)
    print("Random Forest Classifier trained successfully.")

    # Save the trained model
    model_filename = 'models/Online_Retail_model.pkl'
    rf_classifier.save_model(model_filename)

    # Make predictions and evaluate the model
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Accuracy:", accuracy)

    # Load the saved model for future predictions
    loaded_model = RandomForestClassifier.load_model(model_filename)
    # Example usage: loaded_model.predict(X_test)

if __name__ == "__main__":
    main()
