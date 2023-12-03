import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the emotion transformation model
emotion_transformation_model = load_model('emotion_transformation_model.keras')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to apply visual transformation based on the emotion label
def apply_transformation(face_roi, transformed_emotion_label):
    # Example: Rotate the face based on the emotion label
    angle = (transformed_emotion_label - 3) * 10  # Adjust the angle based on your requirements
    center = (face_roi.shape[1] // 2, face_roi.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_face = cv2.warpAffine(face_roi, rotation_matrix, (face_roi.shape[1], face_roi.shape[0]))
    return rotated_face

# Open a video stream (you can adjust the index or file path)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0  # Normalize pixel values
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Use the emotion transformation model to predict the transformed emotion
        transformed_emotion = emotion_transformation_model.predict(face_roi)
        transformed_emotion_label = np.argmax(transformed_emotion)

        # Apply the transformed emotion to the face region
        transformed_face = apply_transformation(face_roi[0], transformed_emotion_label)

        # Resize the transformed face to match the dimensions of the face region
        transformed_face = cv2.resize(transformed_face, (w, h))

        # Replace the original face region with the transformed face
        frame[y:y + h, x:x + w] = transformed_face

    # Display the transformed frame
    cv2.imshow('Transformed Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
