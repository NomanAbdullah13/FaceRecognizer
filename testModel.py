import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained face recognition model
model = load_model('Dataset.keras')

# Define class names for predictions
class_names = ['sabbir', 'noman', 'afif', 'ekram']

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Padding ratio to slightly enlarge the detected face box
padding_ratio = 0.2

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Could not access webcam.")
        break

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Apply padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])

        # Extract the face region
        face_img = frame[y1:y2, x1:x2]

        # Resize to match model input shape
        face_img_resized = cv2.resize(face_img, (128, 128))  # Model expects (128, 128, 3)

        # Normalize and add batch dimension
        face_img_array = np.expand_dims(face_img_resized / 255.0, axis=0)  # Shape: (1, 128, 128, 3)

        # Get predictions from the model
        predictions = model.predict(face_img_array)

        # Get class with the highest probability
        predicted_class = np.argmax(predictions)
        predicted_prob = predictions[0][predicted_class]

        # Display the prediction if confidence is high
        if predicted_prob > 0.9:
            class_name = class_names[predicted_class]
            color = (0, 255, 0)  # Green for confident prediction
        else:
            class_name = 'Unknown'
            color = (0, 0, 255)  # Red for low confidence

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{class_name} ({predicted_prob*100:.2f}%)', 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame with predictions
    cv2.imshow('Face Recognition', frame)

    # Press 'c' to exit
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

# Release webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
