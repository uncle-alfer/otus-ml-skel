import pytest

import pandas as pd

from titanic.features import extract


@pytest.mark.parametrize("test_input,expected", [((0, 0), 1), ((0, 1), 2), ((1, 2), 4)])
def test_family_size(test_input, expected):
    df = pd.DataFrame(
        data={"SibSp": [test_input[0]], "Parch": [test_input[1]]},
    )
    assert expected == extract.family_size(df).iloc[0]


@pytest.mark.parametrize(
    "test_input,expected", [((0, 0), 1), ((0, 1), 0), ((1, 2), 0), ((1, 0), 0)]
)
def test_is_alone(test_input, expected):
    df = pd.DataFrame(
        data={"SibSp": [test_input[0]], "Parch": [test_input[1]]},
    )
    assert expected == extract.is_alone(df).iloc[0]

def test_test_func():
    extract.tst_func()
    assert True
