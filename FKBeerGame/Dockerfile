# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /beer-game-env

COPY . /beer-game-env/

RUN pip install -r requirements.txt

#CMD ["python", "./runner.py"]
CMD ["python", "setup.py", "sdist"]
