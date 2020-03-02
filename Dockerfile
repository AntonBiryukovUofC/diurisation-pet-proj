FROM tensorflow/tensorflow:1.14.0-py3
RUN apt update
RUN apt install gcc musl-dev \
&& pip install cython
RUN apt install -y portaudio19-dev libsndfile1 pulseaudio pavucontrol
ADD requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5000
EXPOSE 5006
RUN mkdir /app
WORKDIR /app
ADD . /app

ENTRYPOINT bash entrypoint.sh