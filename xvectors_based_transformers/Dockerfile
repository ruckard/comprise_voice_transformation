FROM registry.gitlab.inria.fr/comprise/voice_transformation:env

# install the latest upgrades
RUN apt-get update && apt-get -y dist-upgrade

COPY vpc /opt/vpc
COPY app.py /opt
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /opt

CMD ["python3", "app.py"]
