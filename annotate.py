import os
import easyocr

dataset_folder = 'dataset/ocr/'
annotated_folder = 'dataset/ocr_annotated/'

if not os.path.exists(annotated_folder):
    os.makedirs(annotated_folder)

reader = easyocr.Reader(['en'])

for file in os.listdir(dataset_folder):
    # Use easy OCR to extract text from the image
    # print(f'Extracting text from {file}...')
    text = reader.readtext(f'{dataset_folder}{file}')
    # Rename the file to the extracted text
    if text:
        plate_number = text[0][1]
        score = text[0][2]
        if score > 0.55:
            print(f'Extracted text: {plate_number}')
        # os.rename(f'{dataset_folder}{file}',
        #           f'{annotated_folder}{plate_number}.jpg')
    else:
        print(f'No text extracted from {file}')
