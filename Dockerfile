FROM waggle/plugin-base:1.1.1-ml-torch1.9

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir git+https://github.com/ultralytics/ultralytics.git

COPY . .

ENTRYPOINT ["python3", "main.py"]
