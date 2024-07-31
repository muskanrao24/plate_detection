import cv2


def preprocess_image(image):
    # Convert image to grayscale
    gray_plate_region = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    contrast_plate_region = clahe.apply(gray_plate_region)

    return contrast_plate_region
