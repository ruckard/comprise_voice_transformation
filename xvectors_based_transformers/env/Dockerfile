FROM kaldiasr/kaldi:gpu-latest

# install the latest upgrades
RUN apt-get update && apt-get -y dist-upgrade && apt-get -y install cmake python3 python3-pip python3-dev python3-numpy python3-scipy libsndfile1

COPY nii /opt/nii
COPY nii_cmake /opt/nii_cmake
COPY nii_scripts /opt/nii_scripts
COPY install.sh /opt/install.sh
COPY requirements.txt /opt/requirements.txt
# TODO: tmp 
# COPY boost_1_59_0.tar.gz /opt/boost_1_59_0.tar.gz

RUN cd /opt && ./install.sh
ENV KALDI_ROOT=/opt/kaldi
WORKDIR /opt 
