FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git \
 && git clone https://github.com/ultralytics/yolov5.git /app/yolov5 \
 && sed -i '/gitpython/d' /app/yolov5/requirements.txt \
 && sed -i 's/numpy>=1.23.5/numpy<=1.19.5/' /app/yolov5/requirements.txt \
 && sed -i 's/Pillow>=[0-9.]*/Pillow<=9.5.0/' /app/yolov5/requirements.txt \
 && pip3 install --no-cache-dir -r /app/yolov5/requirements.txt \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache/pip

COPY . .

ENTRYPOINT ["python3", "main.py"]
