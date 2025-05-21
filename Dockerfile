FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y git
RUN git clone -b v6.0 https://github.com/ultralytics/yolov5.git /app/yolov5
WORKDIR /app/yolov5
RUN pip3 install -r requirements.txt
WORKDIR /app

COPY . .

ENTRYPOINT ["python3", "main.py"]
