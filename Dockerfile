FROM ubuntu:22.04

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    make cmake unzip git xz-utils\
    curl ca-certificates libcurl4-openssl-dev libssl-dev\
    libgtk2.0-dev libtbb-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 \
    libjpeg-dev libpng-dev libtiff-dev libdc1394-dev \
    libblas-dev libopenblas-dev libeigen3-dev liblapack-dev libatlas-base-dev gfortran \
  && rm -rf /var/lib/apt/lists/*

ARG ZIG_VERSION="0.10.0-dev.4217+9d8cdb855"
ENV ZIG_VERSION ${ZIG_VERSION}
ARG ARCH="x86_64"
ENV ARCH ${ARCH}

RUN curl -q -Lo zig.tar.xz https://ziglang.org/builds/zig-linux-${ARCH}-${ZIG_VERSION}.tar.xz \
  && tar -xvf zig.tar.xz \
  && rm zig.tar.xz \
  && mv zig-linux-${ARCH}-${ZIG_VERSION} zig

ENV CC="/zig/zig cc"
ENV CXX="/zig/zig c++"

ARG OPENCV_VERSION="4.6.0"
ENV OPENCV_VERSION $OPENCV_VERSION

RUN curl -q -Lo opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
  && curl -q -Lo opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
  && unzip opencv.zip \
  && unzip opencv_contrib.zip \
  && mkdir -p opencv-build \
  && cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D WITH_IPP=OFF \
    -D WITH_OPENGL=OFF \
    -D WITH_QT=OFF \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules/ \
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
    ../opencv-${OPENCV_VERSION}/ &&\
   make -j $(nproc --all) \
   && make preinstall \
   && make install \
   && ldconfig \
   && cd / && rm -rf opencv*

