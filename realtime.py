import cv2
import dlib
import joblib
import numpy as np
from sklearn.preprocessing import normalize
import math

# Load the pre-trained SVM model
filename = 'models_SVM.pkl'
models = joblib.load(filename)
clf = models[0]

# Initialize face detector and landmark predictor
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def save_landmarks_to_txt(filename, landmarks_list):
    """
    Save landmarks to a text file.

    Parameters:
    filename (str): Name of the file to save the landmarks.
    landmarks_list (list): List of tuples containing (x, y) coordinates of landmarks.
    """
    with open(filename, 'w') as file:
        for (x, y) in landmarks_list:
            file.write(f"{x} {y}\n")

def draw_landmarks(image, landmarks_list):
    """
    Draw landmarks on the image.

    Parameters:
    image (numpy.ndarray): Input image.
    landmarks_list (list): List of tuples containing (x, y) coordinates of landmarks.
    """
    for (x, y) in landmarks_list:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

def crop_frame(image, landmarks_list, expansion_factor=0.2):
    """
    Crop the frame around the detected landmarks.

    Parameters:
    image (numpy.ndarray): Input image.
    landmarks_list (list): List of tuples containing (x, y) coordinates of landmarks.
    expansion_factor (float): Expansion factor for cropping around landmarks.

    Returns:
    cropped_frame (numpy.ndarray): Cropped frame around the landmarks.
    """
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
    for (x, y) in landmarks_list:
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)

    expansion_x = int((max_x - min_x) * expansion_factor)
    expansion_y = int((max_y - min_y) * expansion_factor)

    min_x -= expansion_x
    min_y -= expansion_y
    max_x += expansion_x
    max_y += expansion_y

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image.shape[1], max_x)
    max_y = min(image.shape[0], max_y)

    cropped_frame = image[min_y:max_y, min_x:max_x]

    return cropped_frame

def normalize_landmarks(landmarks):
    """
    Normalize landmarks to a consistent scale or range.

    Parameters:
    landmarks (numpy.ndarray): Array of shape (68, 2) containing (x, y) coordinates of landmarks.

    Returns:
    normalized_landmarks (numpy.ndarray): Normalized landmarks.
    """
    landmarks = np.array(landmarks)

    # Calculate the centroid (mean) of landmarks
    centroid = landmarks.mean(axis=0)

    # Calculate the distance from each landmark point to the centroid
    distances = np.linalg.norm(landmarks - centroid, axis=1)

    # Compute a scaling factor as the mean distance from the centroid
    scale_factor = distances.mean()

    # Normalize landmarks by scaling each coordinate
    normalized_landmarks = (landmarks - centroid) / scale_factor

    return normalized_landmarks

def main():
    cap = cv2.VideoCapture(0)
    emotions_map = {
        0: "Neutral",
        1: "Sadness",
        2: "Happiness",
        3: "Anger",
        4: "Fear",
        5: "Surprise",
        6: "Contempt",
        7: "Disgust"
    }
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rects:
            # Convert face region to dlib rectangle
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            # Predict landmarks
            shape = predictor(gray, rect)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            # Crop frame around face
            cropped_frame = crop_frame(frame, landmarks)

            # Normalize landmarks
            normalized_landmarks = normalize_landmarks(landmarks)

            # Extract features (example: flatten landmarks)
            features = normalized_landmarks.flatten()

            # Perform prediction
            prediction = clf.predict([features])[0]
            predicted_emotion = emotions_map.get(prediction, "Unknown")

            print(f"Predicted emotion: {predicted_emotion}")

            # Draw landmarks on original frame
            draw_landmarks(frame, landmarks)

            # Display cropped frame and prediction
            cv2.imshow('Cropped Frame', cropped_frame)
            cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display cropped frame and prediction
            # cv2.imshow('Cropped Frame', cropped_frame)
            # cv2.putText(frame, f"Emotion: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
