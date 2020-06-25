FROM nvidia/cuda:10.0-cudnn7-devel
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get -y update && apt-get -y install \
    autoconf \
    automake \
    libtool \
    python-opencv \
    libopencv-dev \
    git \
    wget \
    build-essential \
    cmake \
    python-pip

# pip dependencies
RUN pip install futures grpcio grpcio-tools

RUN git clone https://github.com/AlexeyAB/darknet
WORKDIR /darknet

# compilation
COPY build/Makefile ./
RUN make -j8
