from ultralytics import YOLO

DATASET_CONFIG = ".\dataset\data.yaml"

model = YOLO("yolov8m.pt")

results = model.train(data=DATASET_CONFIG, epochs=1)

# Export the model
model.export()
