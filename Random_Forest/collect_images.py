# Data Collection for Sign Language Recognition
import os
import cv2

# Directory to store the collected data
DATA_DIR = 'Dataset'
# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Total number of sign language classes to collect data for
number_of_classes = 38
# Number of images to collect per class
dataset_size = 100

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    # Create a directory for each class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    
    # Display a message to the user and wait for 'q' press to start capturing
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # Display the frame
        cv2.imshow('frame', frame)
        # Wait for 1 millisecond
        cv2.waitKey(1)
        # Save the frame to the corresponding class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()