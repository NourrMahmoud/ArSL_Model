# Feature Extraction using MediaPipe

# Import necessary libraries
import os
import pickle
import mediapipe as mp   # Used for hand detection and tracking
import cv2               # Used for image processing
import matplotlib.pyplot as plt   # Used for plotting (if needed)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up hand detection with specified parameters
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to the dataset directory
DATA_DIR = 'Dataset'

# Initialize empty lists to store data and labels
data = []
labels = []
# Iterate through each class directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate through each image in the class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Initialize a list to store features for the current image
        data_aux = []
        
        # Initialize lists to store x and y coordinates of hand landmarks
        x_ = []
        y_ = []

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Hands
        results = hands.process(img_rgb)
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Iterate through each hand in the image
            for hand_landmarks in results.multi_hand_landmarks:
                # Iterate through each landmark point of the hand
                for i in range(len(hand_landmarks.landmark)):
                    # Extract x and y coordinates of the landmark
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    # Append the coordinates to the respective lists
                    x_.append(x)
                    y_.append(y)
                
                # Calculate relative coordinates and append to features
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            
            # Append the extracted features to the data list
            data.append(data_aux)
            # Append the corresponding class label to the labels list
            labels.append(dir_)

# Save the extracted data and labels to a pickle file
f = open('data.pickle', 'wb')    # Open a file named 'data.pickle' in write binary ('wb') mode
pickle.dump({'data': data, 'labels': labels}, f)    # Save the data and labels as a dictionary using pickle
f.close()   # Close the file