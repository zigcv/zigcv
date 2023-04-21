FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    make cmake unzip git xz-utils \
    curl ca-certificates libcurl4-openssl-dev libssl-dev \
    libgtk2.0-dev libtbb-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 \
    libjpeg-dev libpng-dev libtiff-dev libdc1394-dev \
    libblas-dev libopenblas-dev libeigen3-dev liblapack-dev libatlas-base-dev gfortran \
  && rm -rf /var/lib/apt/lists/*

ARG ZIG_VERSION="0.10.0-dev.4217+9d8cdb855"
ENV ZIG_VERSION ${ZIG_VERSION}

ARG OPENCV_VERSION="4.6.0"
ENV OPENCV_VERSION $OPENCV_VERSION

WORKDIR /tmp
ENV CC="/tmp/zig/zig cc"
ENV CXX="/tmp/zig/zig c++"
RUN curl -Lso zig.tar.xz https://ziglang.org/builds/zig-linux-$(uname -m)-${ZIG_VERSION}.tar.xz \
  && tar -xf zig.tar.xz \
  && mv zig-linux-$(uname -m)-${ZIG_VERSION} zig \
  && curl -Lso opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
  && curl -Lso opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
  && unzip -qq opencv.zip \
  && unzip -qq opencv_contrib.zip \
  && cd opencv-${OPENCV_VERSION} \
  && mkdir -p build \
  && cd build \
  && cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D WITH_IPP=OFF \
    -D WITH_OPENGL=OFF \
    -D WITH_QT=OFF \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules/ \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_JASPER=OFF \
    -D WITH_TBB=ON \
    -D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_java=NO \
    -D BUILD_opencv_python=NO \
    -D BUILD_opencv_python2=NO \
    -D BUILD_opencv_python3=NO \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    .. \
  && make -j $(nproc --all) \
  && make preinstall \
  && make install \
  && ldconfig \
  && rm -rf /tmp/*

WORKDIR /tmp
ARG ZIGUP_VERSION="v2022_08_25"
RUN curl -Lso zigup.zip https://github.com/marler8997/zigup/releases/download/${ZIGUP_VERSION}/zigup.ubuntu-latest-$(uname -m).zip \
  && unzip -qq zigup.zip \
  && chmod +x zigup \
  && mv zigup /usr/local/bin/zigup \
  && rm -rf /tmp/*

ARG CACHE_DATE=2022-11-09
RUN zigup 0.10.1

