FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models

RUN apt-get update && apt-get install -y wget

RUN wget -q https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx -O /app/models/yolov5s.onnx

COPY coco.names /app/models/coco.names

COPY . .

ENTRYPOINT ["python3", "main.py"]