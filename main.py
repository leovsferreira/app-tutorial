import numpy as np
import cv2
from ultralytics import YOLO
import json

from waggle.plugin import Plugin
from waggle.data.vision import Camera


def detect_objects(image, model):
    results = model(image)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
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
    model = YOLO("yolov8n.pt")
    
    with Plugin() as plugin:
        with Camera("bottom_camera") as camera:
            snapshot = camera.snapshot()
        
        timestamp = snapshot.timestamp
        detections = detect_objects(snapshot.data, model)
        
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detection_data = {
            "detections": detections,
            "counts": class_counts,
            "total_objects": len(detections)
        }
        
        plugin.publish("object.count", len(detections), timestamp=timestamp)
        plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)
        
        img_with_boxes = snapshot.data.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        snapshot.save("snapshot.jpg")
        cv2.imwrite("snapshot_with_detections.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
        plugin.upload_file("snapshot.jpg", timestamp=timestamp)
        plugin.upload_file("snapshot_with_detections.jpg", timestamp=timestamp)


if __name__ == "__main__":
    main()