# Steam Recommendation System Project

End-to-end Steam recommendation system


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [UI](#ui) 


## Background

This recommendation systems is mainly based on [sckit-surpirse](surprise.readthedoc.io) which supports estimating explicit rating data. 
The Steam Game Dataset comes from [Recommender System Datasets](cseweb.ucsd.edu/~jmcauley/datasets/html#steam_data) collected by Julian McAuley at UCSD, and the vote-up(binary) for game by user is taken as explicit rating.

## Install

**Docker**

```bash
docker build -t recdemo .
docker run --rm -p 5000:8080 recdemo
```


**Local development**

To isolate, create virtual environment for local development

```bash
python -m venv .venv
source .venv/bin/activate
cd RecSys
pip install -r requirements.txt
```

Install recsys package
```bash
pip install -e .
```

To confirm installation, get into python terminal
```python
import recsys
```

To exit virtual environment
```bash
deactivate
```

## Usage

Given a pair of userid and itemid, compute esitmated rating 
Given a pair of userid and number of item, compute recommendation list for the userid

Example:
```bash
python inference.py --input_uid "76561198107703934" --input_iid "12210" --input_rec_uid "76561198067243010" --input_n 5
```
Output:
```bash
input user id: 76561198107703934, item id: 12210, estimated rating: 0.8571428571428572
top 5 recommended items for input user id 76561198067243010: ['34010', '204300', '319630', '17390', '47890']
corresponding app name: ['Alpha Protocol™', 'The Sims™ 3', 'Life is Strange - Episode 1', 'Awesomenauts - the 2D moba', 'SPORE™']
```

To get other available userid and itemid, you can call function load_output from from recsys.inference
```python
from recsys.inference import load_output
import pandas as pd
output = load_output()
pred = output["predictions"] #class  surprise.prediction_algorithms.preidction.Prediction
df_pred = pd.DataFrame(pred)
#get the full list of uid and iid
uid = list(df_pred.uid.unique()) 
iid = list(df_pred.iid.unique())
```

## Flask API


**with docker image built**
```
docker run --rm -p 5000:8080 recdemo
```


Start by checking if the api works
```bash
curl -i http://localhost:5000/ping
```

Estimate rating
```bash
curl -X POST -H 'content-type: application/json' --data '{"uid":"76561198107703934","iid":"12210"}' http://localhost:5000/predict
```

Recommend top 5 games for userid
```bash
curl -X POST -H 'content-type: application/json' --data '{"rec_uid":"76561198107703934"}' http://localhost:5000/rec
```

**under virtual environment**
```bash
run main.py
```
the same steps above, except you need to change your IP address to http://0.0.0.0:8000

## UI

Web-based Interactive interface built by Flask and [Dash](https://dash.plot.ly/)((very powerful analytical web applications).

```bash
docker run --rm -p 5000:8080 recdemo
```
```bash
python main.py
```


Go to your browser

http://localhost:5000:8080/ or http://127.0.0.1:8080/

1. Recommender system: gain estimated rating and recommendation list

<image src="https://github.com/jwxu19/steam_game_recommendation_systems/blob/refactor_steam/image/1.png"></image>

http://localhost:5000:8080/dashboard or http://127.0.0.1:8080/dashboard

2. Analytics Dashboard: play around with games and users data with various widgets and graphs


<image src="https://github.com/jwxu19/steam_game_recommendation_systems/blob/refactor_steam/image/2.png"></image>
<p style="text-align: center;">**x-axis**: # of review/recommend for each game, **y-axis**: # of game</p>

<image src="https://github.com/jwxu19/steam_game_recommendation_systems/blob/refactor_steam/image/3.png"></image>
<p style="text-align: center;">**x-axis**: # of review/recommend for each user, **y-axis**: # of user</p>

<image src="https://github.com/jwxu19/steam_game_recommendation_systems/blob/refactor_steam/image/4.png"></image> 
<p style="text-align: center;">**x-axis**: percentage of review/recommend for each user, **y-axis**: # of user</p>



