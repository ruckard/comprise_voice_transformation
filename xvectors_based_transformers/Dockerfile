FROM registry.gitlab.inria.fr/comprise/voice_transformation/env

# install the latest upgrades
RUN apt-get update && apt-get -y dist-upgrade

COPY vpc /opt/vpc
COPY app.py /opt
COPY requirements.txt requirements.txt


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
#RUN conda 

RUN pip3 install -r requirements.txt

WORKDIR /opt

CMD ["python3", "app.py"]
