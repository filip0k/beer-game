# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /beer-game-omg

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /beer-game-omg/

CMD ["python", "./main.py"]
