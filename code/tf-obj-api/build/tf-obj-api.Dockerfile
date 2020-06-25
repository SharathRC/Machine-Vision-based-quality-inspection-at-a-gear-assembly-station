FROM tensorflow/tensorflow:1.15.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get -y update && apt-get -y install \
    # autoconf \
    # automake \
    # libtool \
    python3-opencv \
    libopencv-dev \
    git \
    wget \
    build-essential \
    # cmake \
    python3-pip \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk

# pip dependencies
RUN pip3 install futures grpcio grpcio-tools

# RUN pip3 install tensorflow-gpu

RUN pip3 install \
    Cython \
    contextlib2 \
    matplotlib \
    tf_slim

RUN mkdir -p /tensorflow
WORKDIR /tensorflow

RUN git clone https://github.com/tensorflow/models.git

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.

RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

RUN pip3 install --upgrade tf_slim

RUN python3 setup.py install

RUN cd
RUN mkdir -p /tensorflow/workspace/training_custom
WORKDIR /tensorflow/workspace/training_custom
RUN mkdir annotations
RUN mkdir -p images/test
RUN mkdir -p images/train
RUN mkdir pre-trained-model
RUN mkdir training