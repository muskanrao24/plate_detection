import os
import re
import sqlite3
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
from datetime import datetime
from preprocess import preprocess_image

# International plate detector
# model_path = os.path.join("..", "models", "plate_detector_intl.pt")
# India #1 plate detector
# model_path = os.path.join("..", "models", "plate_detector_india_1.pt")
# India #2 plate detector
# model_path = os.path.join("..", "models", "plate_detector_india_2.pt")
# India #2 plate detector
model_path = os.path.join("..", "models", "plate_detector_india_2_n.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

print("loaded")

# Connect to SQLite database (or create if not exists)
conn = sqlite3.connect("plates.db")

# Create a cursor object to execute SQL commands
cur = conn.cursor()

# Create a table if not exists to store plate numbers and timestamps
cur.execute(
    """CREATE TABLE IF NOT EXISTS plates (
               plate_number TEXT,
               first_seen TEXT,
               last_seen TEXT
               )"""
)

# Commit changes and close connection
conn.commit()


def process_frame(frame):
    global model, threshold, ocr

    H, W, _ = frame.shape

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(
                frame,
                results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )  # License plate text in the box

            # Crop the detected plate region
            plate_region = preprocess_image(frame[int(y1) : int(y2), int(x1) : int(x2)])

            # Perform OCR on the plate region
            result = ocr.ocr(plate_region, det=False)

            if result:
                # Extract the recognized text
                recognized_text, confidence = result[0][0]

                recognized_text = recognized_text.replace(" ", "")

                result_is_plate = re.match(
                    r"^[A-Z]{2}[A-Z|0-9]{2}[A-Z|0-9]{2}[0-9]{4}$", recognized_text
                )

                # Draw the recognized text on the frame
                cv2.putText(
                    frame,
                    recognized_text,
                    (int(x1), int(y2 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                print(result_is_plate, score, recognized_text)
                if result_is_plate and score > 0.45:
                    plate_number = recognized_text
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    conn = sqlite3.connect("plates.db")
                    cur = conn.cursor()

                    # Check if plate number already exists in the database
                    cur.execute(
                        "SELECT * FROM plates WHERE plate_number=?", (plate_number,)
                    )
                    existing_record = cur.fetchone()

                    if existing_record:
                        cur.execute(
                            "UPDATE plates SET last_seen=? WHERE plate_number=?",
                            (timestamp, plate_number),
                        )
                    else:
                        cur.execute(
                            "INSERT INTO plates (plate_number, first_seen, last_seen) VALUES (?, ?, ?)",
                            (plate_number, timestamp, timestamp),
                        )
                    conn.commit()

    return frame