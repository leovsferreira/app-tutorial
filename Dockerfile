# Use a newer Ubuntu base with Python 3.8+
FROM ubuntu:20.04

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

# Install Python 3.8 and required packages
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-distutils \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Install pip for Python 3.8
RUN python3.8 -m pip install --upgrade pip

# Install newer OpenCV and other dependencies
RUN pip3 install opencv-python==4.8.0.74 numpy

# Install waggle dependencies
RUN pip3 install pywaggle[all]==0.56.0

# Install ultralytics
RUN pip3 install ultralytics

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=dialog

WORKDIR /app

# Copy application code
COPY . .

ENTRYPOINT ["python3.8", "main.py"]