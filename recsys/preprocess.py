#!/usr/bin/env python3
import pickle
import json
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join


DATA_DIR = join(dirname(dirname(abspath((__file__)))),
                "data")
JSON_FILE = join(DATA_DIR, "reviews_Movies_and_TV_5.json")
PKL_FILE = join(DATA_DIR, "data_after_filter.pkl")


def load_data(file=JSON_FILE):
    data = []
    for line in open(file, "r"):
        data.append(json.loads(line))

    return data


def save_data(data, file=PKL_FILE):
    outfile = open(file, "wb")
    pickle.dump(data, outfile)
    outfile.close()


def filter_data(data, n=15):
    data = pd.DataFrame.from_dict(data, orient="columns")

    min_n_review_per_movie, min_n_review_per_user = 15, 15

    item_ls_filter = data["asin"].value_counts() > min_n_review_per_movie
    user_ls_filter = data["reviewerID"].value_counts() > min_n_review_per_user
    item_ls_after_filter = item_ls_filter[item_ls_filter].index.tolist()
    user_ls_after_filter = user_ls_filter[user_ls_filter].index.tolist()
    data_after_filter = data[(data["asin"].isin(item_ls_after_filter)) & (
        data["reviewerID"].isin(user_ls_after_filter))]
    data_after_filter = data_after_filter[['reviewerID', 'asin', 'overall']]
    return data_after_filter


def main():
    data = load_data()
    data_after_filter = filter_data(data)
    save_data(data_after_filter)


if __name__ == "__main__":
    main()
