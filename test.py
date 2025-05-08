from ultralytics import YOLO
import cv2
import numpy as np
import os

def test_detection():
    # Load the model
    model = YOLO("runs/detect/train14/weights/best.pt")
    
    # Define the test image path
    test_image_path = "test.png"
    
    # Make sure the test image exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return
    
    # Run detection
    results = model(test_image_path)
    
    # Process results (which is a list of Results objects)
    for i, result in enumerate(results):
        # Load the original image for drawing
        img = cv2.imread(test_image_path)
        
        # Get detection data
        boxes = result.boxes
        
        # Draw rectangles for each detection
        for box in boxes:
            # Get box coordinates (convert to integers for drawing)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence score
            conf = float(box.conf[0])
            
            # Get class id and name
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
            # Draw rectangle
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add label with class name and confidence
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the output image
        output_path = f"/content/yolo/detection_result_{i}.png"
        cv2.imwrite(output_path, img)
        print(f"Saved detection result to {output_path}")
        
        # Print detection data
        print(f"\nDetection {i+1} results:")
        if len(boxes) > 0:
            # Convert to pandas DataFrame for each box separately
            for j, box in enumerate(boxes):
                # Get box coordinates in xywh format
                xywh = box.xywh[0].cpu().numpy()
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                
                print(f"  Object {j+1}: {cls_name}")
                print(f"    Confidence: {conf:.4f}")
                print(f"    Coordinates (xyxy): {xyxy}")
                print(f"    Coordinates (xywh): {xywh}")
        else:
            print("  No objects detected.")

if __name__ == "__main__":
    test_detection()