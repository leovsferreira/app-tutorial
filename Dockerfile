FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && echo "Cloning YOLOv5 master branch..." \
 && git clone https://github.com/ultralytics/yolov5.git /app/yolov5 \
 && echo "--- Original YOLOv5 requirements.txt (relevant lines): ---" \
 && grep -E "numpy|pillow|matplotlib" /app/yolov5/requirements.txt || echo "Relevant lines not found initially." \
 && echo "Modifying YOLOv5 requirements for Python 3.6 compatibility..." \
 && sed -i '/gitpython/d' /app/yolov5/requirements.txt \
 && sed -i 's/^numpy>=1.23.5/numpy==1.19.5/' /app/yolov5/requirements.txt \
 && sed -i 's/^pillow>=10.3.0/pillow==9.5.0/' /app/yolov5/requirements.txt \
 && sed -i 's/^matplotlib>=3.3.0/matplotlib==3.1.3/' /app/yolov5/requirements.txt \
 && sed -i -E 's/^(numpy\s*(>=|~=)\s*)[1-9][0-9]*\.[0-9]+\.[0-9]+(.*)$/numpy==1.19.5\3/' /app/yolov5/requirements.txt \
 && sed -i -E 's/^(pillow\s*(>=|~=)\s*)[1-9][0-9]*\.[0-9]+\.[0-9]+(.*)$/pillow==9.5.0\3/i' /app/yolov5/requirements.txt \
 && sed -i -E 's/^(matplotlib\s*(>=|~=)\s*)[3-9]\.[0-9]+(\.[0-9]+)?(.*)$/matplotlib==3.1.3\4/' /app/yolov5/requirements.txt \
 && echo "--- Modified YOLOv5 requirements.txt (relevant lines): ---" \
 && grep -E "numpy|pillow|matplotlib" /app/yolov5/requirements.txt || echo "Relevant lines not found after modification." \
 && echo "--- Full Modified YOLOv5 requirements.txt for pip: ---" \
 && cat /app/yolov5/requirements.txt \
 && echo "Installing modified YOLOv5 requirements..." \
 && pip3 install --no-cache-dir -r /app/yolov5/requirements.txt \
 && echo "Cleaning up apt cache..." \
 && rm -rf /var/lib/apt/lists/* \
 && echo "Cleaning up pip cache..." \
 && rm -rf /root/.cache/pip

COPY . .
ENTRYPOINT ["python3", "main.py"]
