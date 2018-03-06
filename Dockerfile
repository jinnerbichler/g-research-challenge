FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

LABEL maintainer="Craig Citro <craigcitro@google.com>"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-numpy \
        python3-dev \
        python3-pip \
        python3-wheel \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install and update pip tools
RUN pip3 install --upgrade pip
RUN pip3 install setuptools

# Install dependencies
RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        && \
    python3 -m ipykernel.kernelspec

# Install TensorFlow GPU version.
RUN pip3 install tensorflow-gpu==1.6.0

COPY . .

RUN pip3 install -r requirements-gpu.txt

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

EXPOSE 6006

ENTRYPOINT ["/usr/bin/python3", "main.py"]