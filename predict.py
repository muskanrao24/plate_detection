import os
import re
import sqlite3
import cv2
import easyocr
from ultralytics import YOLO
from datetime import datetime

# Define paths
model_path = os.path.join(".", "models", "plate_detector.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Connect to SQLite database (or create if not exists)
conn = sqlite3.connect('plates.db')

# Create a cursor object to execute SQL commands
cur = conn.cursor()

# Create a table if not exists to store plate numbers and timestamps
cur.execute('''CREATE TABLE IF NOT EXISTS plates (
               plate_number TEXT,
               first_seen TEXT,
               last_seen TEXT
               )''')

# Commit changes and close connection
conn.commit()

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  # License plate text in the box

            # Crop the detected plate region
            plate_region = frame[int(y1):int(y2), int(x1):int(x2)]

            # Perform OCR on the plate region
            result = reader.readtext(plate_region)

            if result:
                # Extract the recognized text
                recognized_text = result[0][1]

                result_is_plate = re.match(
                    r'^[A-Z]-[A-Z|0-9]{3}-[A-Z|0-9]{2}$', recognized_text)

                # Draw the recognized text on the frame
                cv2.putText(frame, recognized_text, (int(x1), int(y2 + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if result_is_plate and score > 0.95:
                    plate_number = recognized_text
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Check if plate number already exists in the database
                    cur.execute("SELECT * FROM plates WHERE plate_number=?",
                                (plate_number,))
                    existing_record = cur.fetchone()

                    if existing_record:
                        cur.execute(
                            "UPDATE plates SET last_seen=? WHERE plate_number=?", (timestamp, plate_number))
                    else:
                        cur.execute("INSERT INTO plates (plate_number, first_seen, last_seen) VALUES (?, ?, ?)",
                                    (plate_number, timestamp, timestamp))
                    conn.commit()

    # Write the frame into the output video
    output_video.write(frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam, VideoWriter, and close OpenCV windows
cap.release()
output_video.release()
cv2.destroyAllWindows()
