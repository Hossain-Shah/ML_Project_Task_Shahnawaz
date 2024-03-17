import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model_development import RandomForestClassifier, preprocess_data, load_csv

# Load data from CSV file or use existing data
filename = 'data/Online_Retail_Data_Set.csv'
dataset, header = load_csv(filename)
print("Dataset loaded successfully.")
print("Header:", header)

# Preprocess data
X_test, y_test = preprocess_data(np.array(dataset))

# Load the saved model
loaded_model = RandomForestClassifier.load_model('models/Online_Retail_model.pkl')

# Make predictions on the test set
y_pred_test = loaded_model.predict(X_test)

# Calculate accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Set Accuracy:", accuracy_test)

# Generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Generate and display classification report
report = classification_report(y_test, y_pred_test)
print("Classification Report - Test Set:")
print(report)
