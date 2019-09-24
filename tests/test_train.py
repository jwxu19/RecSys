import pytest
from recsys.train import *

@pytest.mark.parametrize("loaded_data, length",
                         [pytest.param(load_filtered_data(), 787544)])
def test_load_filtered_data(loaded_data, length):
    assert length == len(loaded_data)
