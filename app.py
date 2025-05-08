from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")  # You can use a different model like yolov8s.pt, yolov8m.pt, etc.

# Train the model on your custom dataset
results = model.train(
    data="datasets/object_dataset/dataset.yaml",  # Path to your dataset YAML
    epochs=100,  # Number of epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size (adjust based on your GPU memory)
)

# After training, run inference on a test image
results = model.predict("test.png")

# Print the results
print(results.pandas().xywh)  # Shows the predictions in a pandas DataFrame
