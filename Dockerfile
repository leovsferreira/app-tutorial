FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git \
 && git clone https://github.com/ultralytics/yolov5.git /app/yolov5 \
 && sed -i '/gitpython/d' /app/yolov5/requirements.txt \
 && pip3 install --no-cache-dir -r /app/yolov5/requirements.txt \
 && rm -rf /var/lib/apt/lists/*

COPY . .

ENTRYPOINT ["python3", "main.py"]
