import numpy as np
import cv2
from ultralytics import YOLO
import json

from waggle.plugin import Plugin
from waggle.data.vision import Camera


def detect_objects(image, model):
    # Run YOLOv8 inference on the image
    results = model(image)
    
    # Get detected objects
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class, confidence, and box coordinates
            cls = int(box.cls.item())
            cls_name = model.names[cls]
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
    
    return detections


def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")
    
    with Plugin() as plugin:
        # Open camera and take snapshot
        with Camera("bottom_camera") as camera:
            snapshot = camera.snapshot()
        
        # Get timestamp for consistent timing across measurements
        timestamp = snapshot.timestamp
        
        # Detect objects in the image
        detections = detect_objects(snapshot.data, model)
        
        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        # Create a combined JSON with both detections and counts
        detection_data = {
            "detections": detections,
            "counts": class_counts,
            "total_objects": len(detections)
        }
        
        # Publish detection results
        plugin.publish("object.count", len(detections), timestamp=timestamp)
        plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)
        
        # Draw bounding boxes and save images
        img_with_boxes = snapshot.data.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save and upload images
        snapshot.save("snapshot.jpg")
        cv2.imwrite("snapshot_with_detections.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        plugin.upload_file("snapshot.jpg", timestamp=timestamp)
        plugin.upload_file("snapshot_with_detections.jpg", timestamp=timestamp)


if __name__ == "__main__":
    main()