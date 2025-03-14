# Real-time Sign Language Translation

# Import necessary libraries
import pickle   # For loading the trained model
import cv2      # For image processing and video capture
import mediapipe as mp   # For hand detection and tracking
import numpy as np       # For numerical operations

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']     # Extract the model from the dictionary

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up hand detection with specified parameters
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping class indices to corresponding sign language labels
labels_dict = {0: 'Alef', 1: 'Ba2', 2: 'Ta2', 3: 'Tha2', 4: 'Geem',
               5: '7a2', 6: 'Kha2', 7: 'Daal', 8: 'Zaal', 9: 'Ra2',
               10: 'Zeen', 11: 'Seen', 12: 'Sheen', 13: 'Sad',
               14: 'Dad', 15: 'Tah', 16: 'Zah', 17: 'Ain', 18: 'Ghain',
               19: 'Fa2', 20: 'Qaf', 21: 'Kaf', 22: 'Lam',
               23: 'Meem', 24: 'Noon', 25: 'Heeh', 26: 'Waaw', 27: 'Ya2',
               28: '0', 29: '1', 30: '2', 31: '3', 32: '4', 33: '5',
               34: '6', 35: '7', 36: '8', 37: '9'}

# Main loop for real-time translation
while True:
    # Initialize lists to store hand landmark data
    data_aux = []
    x_ = []
    y_ = []
    
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Get frame dimensions
    H, W, _ = frame.shape
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Iterate through each hand in the image
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Extract hand landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)
            
            # Calculate relative coordinates and append to features
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Padding for consistent feature size
        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))
        
        # Calculate bounding box coordinates around the hand
        x1 = int(min(x_) * W) - 10   # x-coordinate of the top-left corner
        y1 = int(min(y_) * H) - 10   # y-coordinate of the top-left corner
        x2 = int(max(x_) * W) - 10   # x-coordinate of the bottom-right corner
        y2 = int(max(y_) * H) - 10   # y-coordinate of the bottom-right corner

        # Make prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])
        
        # Get the predicted character from the labels dictionary
        predicted_character = labels_dict[int(prediction[0])]
        
        # Draw bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw bounding box
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)     # Display predicted character

    # Display the frame with hand landmarks and prediction
    cv2.imshow('Sign Translation', frame)
    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break


# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()