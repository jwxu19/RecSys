import pytest
from surprise import KNNWithMeans, SVD, SVDpp, SlopeOne, CoClustering
from surprise import Reader, Dataset
from recsys.train import load_data, iterate_algo, save_output, set_seed, save_output, find_best_model, refit
from os.path import abspath, dirname, join

DATA_DIR = join(dirname(dirname(abspath((__file__)))), "data")
PKL_FILE = join(DATA_DIR, "data_after_filter.pkl")
METRICS_FILE = join(DATA_DIR, "metrics.pkl")
TEST_METRICS_FILE = join(DATA_DIR, "test_metrics.pkl")
TEST_OUTPUT_FILE = join(DATA_DIR, "test_best_model_predictions.pkl")
TEST_TEMP_METRICS_FILE = join(DATA_DIR, "test_temp_metrics.pkl")
TEST_TEMP_OUTPUT_FILE = join(DATA_DIR, "test_temp_output.pkl")


def test_load_data():
    assert len(load_data(PKL_FILE)) == 787544


def test_iterate_algo():
    set_seed()
    data = load_data(PKL_FILE)
    data = Dataset.load_from_df(data, reader=Reader(rating_scale=(1, 5)))
    kfold = 5
    algo_ls = (KNNWithMeans(), SVDpp())
    top_n = 10
    threshold = 4
    k_ls = [3, 5, 7, 10]
    metrics = iterate_algo(algo_ls, kfold, data, top_n, threshold, k_ls)
    save_output(metrics, TEST_TEMP_METRICS_FILE)
    expected_metrics = load_data(TEST_METRICS_FILE)
    loaded_temp_metrics = load_data(TEST_TEMP_METRICS_FILE)
    assert metrics == expected_metrics
    assert metrics == loaded_temp_metrics


def test_find_best_model():
    #    algo_ls = (KNNWithMeans(), SVDpp())
    algo_ls = (KNNWithMeans(),
               SVD(), SVDpp(),
               SlopeOne(), CoClustering())
    metrics = load_data(METRICS_FILE)
    algo_dict = dict(zip(metrics["algo_name"], algo_ls))
    best_algo_name = find_best_model(algo_dict, metrics)
    assert best_algo_name == "SVDpp "


def test_refit():
    set_seed()
    data = load_data(PKL_FILE)
    data = Dataset.load_from_df(data, reader=Reader(rating_scale=(1, 5)))
    algo_ls = (KNNWithMeans(), SVDpp())
    metrics = load_data(TEST_METRICS_FILE)
    algo_dict = dict(zip(metrics["algo_name"], algo_ls))
    output = refit(data, algo_dict[find_best_model(algo_dict, metrics)])
    save_output(output, TEST_TEMP_OUTPUT_FILE)
    expected_output = load_data(TEST_OUTPUT_FILE)
    loaded_temp_output = load_data(TEST_TEMP_OUTPUT_FILE)
    assert output["predictions"] == expected_output["predictions"]
    assert output["algo"] == expected_output["algo"]
    assert output == loaded_temp_output
