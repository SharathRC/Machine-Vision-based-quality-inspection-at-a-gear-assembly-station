FROM tensorflow/tensorflow:2.0.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# # Add new user to avoid running as root
# RUN useradd -ms /bin/bash tensorflow
# USER tensorflow
WORKDIR /home/tensorflow

# # Copy this version of of the model garden into the image
# COPY --chown=tensorflow . /home/tensorflow/models
RUN git clone https://github.com/tensorflow/models.git

# Compile protobuf configs
WORKDIR /home/tensorflow/models/research/
 
RUN protoc object_detection/protos/*.proto --python_out=.

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

RUN cd
RUN mkdir -p /tensorflow/workspace/training_custom
WORKDIR /tensorflow/workspace/training_custom

ENV TF_CPP_MIN_LOG_LEVEL 3