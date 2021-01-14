FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN pip install azureml-defaults tensorflow-gpu==2.2.0 absl-py Pillow

RUN mkdir /install
RUN mkdir /install/TensorFlow

WORKDIR /install/TensorFlow

RUN wget https://github.com/tensorflow/models/archive/master.zip
RUN unzip master.zip
RUN mv models-master/ models/

RUN mkdir /install/proto
WORKDIR /install/proto
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
RUN unzip protoc-3.13.0-linux-x86_64.zip
ENV PATH="${PATH}:/install/proto/bin"

WORKDIR /install/TensorFlow/models/research
RUN /install/proto/bin/protoc object_detection/protos/*.proto --python_out=.

RUN cp object_detection/packages/tf2/setup.py .
RUN python -m pip install .

# FINISH
WORKDIR /