FROM tensorflow/tensorflow:2.1.0-py3
LABEL maintainer="Lorenzo Rizzello <iotwithit@gmail.com>"

RUN pip install matplotlib \
                tensorflow-datasets
RUN apt-get -y update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:team-gcc-arm-embedded/ppa && \
    apt-get -y update && \
    apt-get -y install gcc-arm-embedded && \
    apt-get -y install libc6-i386
RUN apt-get -y install git

