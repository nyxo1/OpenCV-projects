import cv2
import os

# Create folder for saved images
output_folder = "saved_smiles"
os.makedirs(output_folder, exist_ok=True)

# Load Haar cascades
face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
smile_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

smile_count = 1
was_smiling = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = face_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        # Detect smiles within face
        smiles = smile_model.detectMultiScale(face_gray, 1.8, 20)

        # Draw blue rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Check if smiling
        if len(smiles) > 0:
            label = "Smiling :)"
            color = (0, 255, 0)

            # Saving the frame with smile
            if not was_smiling:
                filename = f"{output_folder}/smile{smile_count}.jpg"
                cv2.imwrite(filename, face_color)
                smile_count += 1

            was_smiling = True

        else:
            label = "No smile"
            color = (0, 0, 255)
            was_smiling = False

        # Put label above face
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face and Smile Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()