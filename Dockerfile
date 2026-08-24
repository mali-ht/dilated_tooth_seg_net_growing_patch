# Base image must match requirements.txt's torch +cu128 (see Docs/SETUP.md). Bumped 2026-08-22
# from nvidia/cuda:12.1.0-devel-ubuntu20.04 alongside the Blackwell/sm_120 migration; a cu121 base
# under a cu128 torch is exactly the mismatch that builds fine and dies at the first kernel launch.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# Add the deadsnakes PPA for Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    python3.11-venv \
    apt-utils \
    wget \
    git

# Update alternatives to set python3 to point to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# alias python='python3'
RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/

RUN pip install -r requirements.txt

# jupyterlab is installed HERE, not in requirements.txt, because only this image's CMD needs it.
# The 2026-08-22 dependency prune removed it (and 15 other never-imported packages) from
# requirements.txt; installing it here keeps this image's default CMD working without putting a
# notebook stack back into every server install.
RUN pip install ninja jupyterlab


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
EXPOSE 6006