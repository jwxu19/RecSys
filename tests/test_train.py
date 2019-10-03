from os.path import abspath, dirname, join
import pytest
from surprise import KNNWithMeans, SVD, SVDpp, SlopeOne, CoClustering
from surprise import Reader, Dataset
import pandas as pd
from recsys.train import (load_data, iterate_algo,
                          set_seed, save_output, find_best_model, refit)


DATA_DIR = join(dirname(dirname(abspath((__file__)))), "data")
PKL_FILE = join(DATA_DIR, "data_after_filter.pkl")

TEST_DATA_DIR = join(dirname(abspath((__file__))), "validation_data")
TEST_DATA_FILE = join(TEST_DATA_DIR, "test_data.pkl")
TEST_METRICS_FILE = join(TEST_DATA_DIR, "test_metrics.pkl")
TEST_OUTPUT_FILE = join(TEST_DATA_DIR, "test_best_model_predictions.pkl")

TEST_TEMP_METRICS_FILE = join(TEST_DATA_DIR, "test_temp_metrics.pkl")
TEST_TEMP_OUTPUT_FILE = join(TEST_DATA_DIR, "test_temp_output.pkl")


def test_load_data():
    data = load_data(PKL_FILE)
    test_data = load_data(TEST_DATA_FILE)
    assert len(data) == 787544
    assert isinstance(data, pd.DataFrame)
    assert len(test_data) == 5773
    assert isinstance(test_data, pd.DataFrame)


def test_iterate_algo():
    set_seed()
    data = load_data(TEST_DATA_FILE)
    data = Dataset.load_from_df(data, reader=Reader(rating_scale=(1, 5)))
    kfold = 5
    algo_ls = (KNNWithMeans(),
               SVD(), SVDpp(),
               SlopeOne(), CoClustering())
    top_n = 10
    threshold = 3.5
    k_ls = [3, 5, 7, 10]
    metrics = iterate_algo(algo_ls, kfold, data, top_n, threshold, k_ls)
    save_output(metrics, TEST_TEMP_METRICS_FILE)
    expected_metrics = load_data(TEST_METRICS_FILE)
    loaded_temp_metrics = load_data(TEST_TEMP_METRICS_FILE)
    assert metrics.keys() == expected_metrics.keys()
    assert metrics["cv_rmse"] == expected_metrics["cv_rmse"]
    assert metrics["cv_precision"] == expected_metrics["cv_precision"]
    assert metrics["cv_recall"] == expected_metrics["cv_recall"]
    assert metrics["cv_personalization"] == expected_metrics["cv_personalization"]
    assert metrics["algo_name"] == expected_metrics["algo_name"]
    assert metrics == loaded_temp_metrics


def test_find_best_model():
    algo_ls = (KNNWithMeans(),
               SVD(), SVDpp(),
               SlopeOne(), CoClustering())
    metrics = load_data(TEST_METRICS_FILE)
    algo_dict = dict(zip(metrics["algo_name"], algo_ls))
    best_algo_name = find_best_model(algo_dict, metrics)
    assert best_algo_name == "SVDpp "


def test_refit():
    set_seed()
    data = load_data(TEST_DATA_FILE)
    data = Dataset.load_from_df(data, reader=Reader(rating_scale=(1, 5)))
    output = refit(data, SVDpp())
    save_output(output, TEST_TEMP_OUTPUT_FILE)
    expected_output = load_data(TEST_OUTPUT_FILE)
    loaded_temp_output = load_data(TEST_TEMP_OUTPUT_FILE)
    assert output["predictions"] == loaded_temp_output["predictions"]
    assert output["predictions"] == expected_output["predictions"]
