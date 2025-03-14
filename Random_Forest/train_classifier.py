# Model Training and Evaluation

# Import necessary libraries
import pickle    # For loading the data
import numpy as np   # For numerical operations
from sklearn.ensemble import RandomForestClassifier    # For model training
from sklearn.model_selection import train_test_split   # For data splitting
from sklearn.metrics import accuracy_score             # For model evaluation

# Load the data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels from the dictionary
data = data_dict['data']
labels = np.asarray(data_dict['labels'])   # Convert labels to a NumPy array

# Data Preprocessing: Padding

# Find the maximum length of the inner lists (feature vectors)
max_length = max(len(row) for row in data)

# Pad the shorter lists with zeros to make them equal in length
# This ensures consistent feature vector size for the model
padded_data = [row + [0] * (max_length - len(row)) for row in data]

# Convert the padded data to a NumPy array
data = np.array(padded_data)


# Model Training and Evaluation

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# test_size=0.2 means 20% of the data will be used for testing
# shuffle=True shuffles the data before splitting
# stratify=labels ensures that the proportion of classes is maintained in both sets

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_predict = model.predict(x_test)

# Evaluate the model's accuracy
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file
f = open('model.p', 'wb')     # Open a file named 'model.p' in write binary ('wb') mode
pickle.dump({'model': model}, f)   # Save the trained model to the file using pickle
f.close()   # Close the file