FROM python:3.8-slim as python_base

RUN pip install ultralytics opencv-python numpy

FROM waggle/plugin-base:1.1.1-ml

COPY --from=python_base /usr/local/bin/python3.8 /usr/local/bin/python3.8
COPY --from=python_base /usr/local/lib/python3.8 /usr/local/lib/python3.8
COPY --from=python_base /usr/local/lib/libpython3.8.so* /usr/local/lib/
COPY --from=python_base /usr/local/include/python3.8 /usr/local/include/python3.8

RUN ln -sf /usr/local/bin/python3.8 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.8 /usr/local/bin/python

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3.8", "main.py"]