FROM ubuntu:18.04

LABEL maintainer="AI2Life"
LABEL version="0.1"
LABEL description="minichatbot on Ubuntu 18.04"

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND noninteractive 

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y nmap\
	libopencv-dev \
    python3-pip \
    python3.6 \
    python3-dev \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip &&\
    pip3 install --upgrade numpy && \ 
    pip3 install --upgrade pandas && \  
    pip3 install --upgrade sklearn && \  
    pip3 install --upgrade matplotlib && \  
    pip3 install --upgrade seaborn && \  
    pip3 install --upgrade pyyaml && \  
    pip3 install --upgrade h5py && \ 
    pip3 install --upgrade tensorflow-gpu && \
    pip3 install --upgrade keras && \
    pip3 install --upgrade opencv-python && \
    pip3 install --upgrade imutils && \
    pip3 install --upgrade Flask && \
    pip3 install --upgrade Flask-SocketIO && \
    pip3 install --upgrade nltk &&\
    pip3 install --upgrade sh &&\
    pip3 install --upgrade gevent-websocket      

RUN ["mkdir", "minibot"]
WORKDIR /minibot 
COPY . /minibot

EXPOSE  8089

VOLUME /minibot

CMD [ "python3", "application.py"]