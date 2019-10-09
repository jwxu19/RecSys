FROM python:3.6

WORKDIR /steam_game_recommendation_systems
COPY . /setam_game_recommendation_systems

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

RUN pip install -r requirements.txt

EXPOSE 5000
CMD python main.py
