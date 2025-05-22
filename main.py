import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import cv2
import torch
import json

from waggle.plugin import Plugin
from waggle.data.vision import Camera

def main():
    device = 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=device)
    model.conf = 0.2
    model.iou = 0.45 
    
    with Plugin() as plugin:
        with Camera() as camera:
            snapshot = camera.snapshot()
        
        timestamp = snapshot.timestamp
        
        results = model(snapshot.data)
        
        detections = []
        results_data = results.pandas().xyxy[0]
        
        for _, row in results_data.iterrows():
            detections.append({
                "class": row['name'],
                "confidence": float(row['confidence']),
                "bbox": [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
            })
        
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
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