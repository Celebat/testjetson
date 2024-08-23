FROM nvcr.io/nvidia/l4t-base:r32.4.4

RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

COPY yolo_rtsp.py /workspace/yolo_rtsp.py

WORKDIR /workspace

CMD ["python3", "yolo_rtsp.py"]
