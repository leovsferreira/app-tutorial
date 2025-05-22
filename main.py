import numpy as np
import cv2
import json

from waggle.plugin import Plugin
from waggle.data.vision import Camera


def main():
    net = cv2.dnn.readNetFromONNX('/app/models/yolov5s.onnx')
    
    with open('/app/models/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print("Model loaded successfully")
    
    with Plugin() as plugin:
        with Camera() as camera:
            snapshot = camera.snapshot()
        
        timestamp = snapshot.timestamp
        
        img = snapshot.data.copy()
        height, width = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        outputs = net.forward()
        
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    detections.append({
                        "class": classes[class_id],
                        "confidence": float(confidence),
                        "bbox": [x, y, x + w, y + h]
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
        
        plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)
        
        img_with_boxes = img.copy()
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