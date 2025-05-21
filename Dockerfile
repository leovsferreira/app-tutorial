FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install torch==1.7.1 torchvision==0.8.2

RUN apt-get update && apt-get install -y git
RUN git clone -b v6.0 https://github.com/ultralytics/yolov5.git /app/yolov5
WORKDIR /app/yolov5
RUN pip3 install -r requirements.txt
WORKDIR /app
COPY . .

ENTRYPOINT ["python3", "main.py"]
