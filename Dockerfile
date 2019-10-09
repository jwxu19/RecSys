FROM python:3.6


#ENV PATH="/steam_game_recommendation_systems:${PATH}"
#WORKDIR /steam_game_recommendation_systems
COPY . /
COPY requirements.txt ./
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "./main.py"]
