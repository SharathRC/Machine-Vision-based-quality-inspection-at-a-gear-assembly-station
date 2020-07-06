FROM tensorflow/tensorflow:2.0.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get -y update && apt-get -y install \
    # autoconf \
    # automake \
    # libtool \
    python3-pip \
    # python3-opencv \
    libopencv-dev \
    git \
    wget \
    build-essential \
    # cmake \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk

RUN python3 -m pip install --upgrade pip

# pip dependencies
RUN pip3 install futures grpcio grpcio-tools

# RUN pip3 install tensorflow-gpu

RUN pip3 install --upgrade \
    Cython \
    contextlib2 \
    matplotlib \
    tf_slim \
    numpy \
    opencv-python \
    tensorflow_datasets \
    scipy