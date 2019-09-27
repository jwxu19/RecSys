from os.path import abspath, dirname, join
import pytest
from surprise import KNNWithMeans, SVD, SVDpp, SlopeOne, CoClustering
from surprise import Reader, Dataset
import pandas as pd
from recsys.evaluate import get_top_n, personalization

DATA_DIR = join(dirname(dirname(abspath((__file__)))), "data")

TEST_METRICS_FILE = join(DATA_DIR, "test_metrics.pkl")
TEST_OUTPUT_FILE = join(DATA_DIR, "test_best_model_predictions.pkl")

TEST_TEMP_METRICS_FILE = join(DATA_DIR, "test_temp_metrics.pkl")
TEST_TEMP_OUTPUT_FILE = join(DATA_DIR, "test_temp_output.pkl")

predictions = [["a", "1", 4, 4.1339, "details"],
               ["a", "2", 3, 2.9187, "details"],
               ["a", "5", 4, 3.1339, "details"],
               ["b", "1", 3, 3.3971, "details"],
               ["b", "7", 4, 4.1339, "details"],
               ["c", "2", 3, 2.1339, "details"],
               ["c", "3", 4, 4.1339, "details"],
               ["d", "6", 2, 1.3981, "details"]]


def test_get_top_n():
    assert get_top_n(predictions, 1) == {'a': [('1', 4.1339)],
                                         'b': [('7', 4.1339)],
                                         'c': [('3', 4.1339)],
                                         'd': [('6', 1.3981)]}


def test_personalization():
    assert round(personalization(predictions, 2), 5) == 0.91667
