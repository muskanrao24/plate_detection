import os
import re
import sqlite3

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
from datetime import datetime


# Define paths
TEST_DIR = os.path.join(".", "test")
video_path = os.path.join(TEST_DIR, "demo.mp4")
video_path_out = f'{video_path}_out.mp4'
model_path = os.path.join(".", "models", "plate_detector.pt")

# Initialize Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

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

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(
    *'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Crop the detected plate region
            plate_region = frame[int(y1):int(y2), int(x1):int(x2)]

            # Convert plate region to grayscale
            gray_plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

            # Perform OCR on the plate region
            result = ocr.ocr(gray_plate_region, det=False)

            if result:
                # Extract the recognized text
                recognized_text, confidence = result[0][0]

                recognized_text = recognized_text.replace(" ", "")

                result_is_plate = re.match(
                    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', recognized_text)

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
                    # Add plate number to log string
                    log_string += f"Plate: {plate_number}, "

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
