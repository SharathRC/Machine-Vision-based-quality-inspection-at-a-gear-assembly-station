FROM nvidia/cuda:10.0-cudnn7-devel
# FROM nvidia/cuda:9.0-cudnn7-devel
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get -y update && apt-get -y install \
    python3.6 \
    autoconf \
    automake \
    libtool \
    python3-opencv \
    libopencv-dev \
    git \
    wget \
    build-essential \
    cmake \
    python3-pip

RUN python3 -m pip install --upgrade pip

# pip dependencies
RUN pip3 install futures grpcio grpcio-tools

RUN git clone https://github.com/AlexeyAB/darknet
WORKDIR /darknet

# compilation
COPY build/Makefile ./
RUN make -j8
