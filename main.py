import json
import cv2
import torch

from waggle.plugin import Plugin
from waggle.data.vision import Camera


def detect_objects(image, model):
    results = model(image)
    detections = []

    for *xyxy, conf, cls in results.xyxy[0].tolist():
        cls = int(cls)
        cls_name = model.names[cls]
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append({
            "class":      cls_name,
            "confidence": conf,
            "bbox":       [x1, y1, x2, y2]
        })
    return detections


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.conf = 0.15

    with Plugin() as plugin:
        with Camera("bottom_camera") as camera:
            snapshot = camera.snapshot()

        ts = snapshot.timestamp
        img = snapshot.data

        detections = detect_objects(img, model)

        counts = {}
        for det in detections:
            counts[det["class"]] = counts.get(det["class"], 0) + 1

        payload = {
            "detections":    detections,
            "counts":        counts,
            "total_objects": len(detections)
        }

        plugin.publish("object.count", len(detections), timestamp=ts)
        plugin.publish("object.detections", json.dumps(payload), timestamp=ts)

        img_boxes = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"{det['class']}:{det['confidence']:.2f}"
            cv2.putText(img_boxes, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        snapshot.save("snapshot.jpg")
        cv2.imwrite("snapshot_with_detections.jpg",
                    cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR))

        plugin.upload_file("snapshot.jpg", timestamp=ts)
        plugin.upload_file("snapshot_with_detections.jpg", timestamp=ts)


if __name__ == "__main__":
    main()