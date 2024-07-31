import os
import cv2
from preprocess import preprocess_image
from paddleocr import PaddleOCR
from ultralytics import YOLO
import xml.etree.ElementTree as ET

ocr = PaddleOCR(use_angle_cls=True, lang='en')


def extract_plate_numbers(xml_dir):
    plate_numbers_dict = {}
    for txt_file in os.listdir(xml_dir):
        if txt_file.endswith(".txt"):
            with open(os.path.join(xml_dir, txt_file), "r") as file:
                plate_numbers = file.readlines()
                plate_numbers = [plate_number.strip()
                                 for plate_number in plate_numbers]
                plate_numbers_dict[txt_file.replace(
                    ".txt", "")] = plate_numbers
    return plate_numbers_dict


# Path to YOLO dataset's train images
test_images_dir = "../dataset/ocr/images"
test_labels_dir = "../dataset/ocr/labels"

# Path to YOLO model
model_path = "../models/plate_detector_india_2.pt"

# Load YOLO model
model = YOLO(model_path)

plate_numbers_dict = extract_plate_numbers(test_labels_dir)

# Load and preprocess images
total_images = 0
correct_predictions = 0
for filename in os.listdir(test_images_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(test_images_dir, filename)
        image = cv2.imread(image_path)

        results = model(image)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.25:
                # Crop the detected plate region
                plate_region = preprocess_image(
                    image[int(y1):int(y2), int(x1):int(x2)])

                # Perform OCR on the plate region
                result = ocr.ocr(plate_region, det=False)

                if result:
                    # Extract the recognized text
                    recognized_text, confidence = result[0][0]
                    # Replace all ., - and spaces with empty string
                    recognized_text = recognized_text.replace(
                        ".", "").replace("-", "").replace(" ", "")
                    recognized_text = recognized_text.upper()

                    # Compare predicted plate number with ground truth
                    xml_filename = filename.replace(".jpg", "")

                    print("Predicted:", recognized_text, "Actual:",
                          plate_numbers_dict.get(xml_filename, []))
                    if recognized_text in plate_numbers_dict.get(xml_filename, []):
                        correct_predictions += 1

                    total_images += 1

# Calculate accuracy
accuracy = correct_predictions / total_images if total_images > 0 else 0
print("Accuracy:", accuracy)
