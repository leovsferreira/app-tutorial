FROM waggle/plugin-base:1.1.1-ml-torch1.9.0

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install ultralytics

COPY . .

ENTRYPOINT ["python3", "main.py"]