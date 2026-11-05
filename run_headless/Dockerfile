FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY docker_host_ca.crt /usr/local/share/ca-certificates/docker-host-ca.crt

RUN mkdir -p /etc/ssl/certs \
    && cat /usr/local/share/ca-certificates/docker-host-ca.crt >> /etc/ssl/certs/ca-certificates.crt \
    && sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        file \
        git \
        libgl1 \
        libglu1-mesa \
        libopencl1 \
        libqt5core5a \
        libqt5gui5 \
        libqt5network5 \
        libqt5opengl5 \
        libqt5widgets5 \
        libqt5xml5 \
        libx11-6 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxext6 \
        libxi6 \
        libxkbcommon-x11-0 \
        make \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        python3-wheel \
        x11-apps \
        xdotool \
        xvfb \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        h5py \
        matplotlib \
        numpy \
        pandas \
        plotly \
        scipy

ENV PYTHONUNBUFFERED=1
ENV PYTHON_EXE=/usr/bin/python3
ENV QT_QPA_PLATFORM=offscreen
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/xdg-cache
ENV OMP_NUM_THREADS=4

WORKDIR /work/VorLap
