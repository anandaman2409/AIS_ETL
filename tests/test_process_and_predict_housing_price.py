from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from etl.constants import CATEGORICAL_COLUMN
from etl.exceptions import (
    CategoricalColumnNotFoundException,
    PredictColumnNotFoundException,
    InvalidInputException,
)
from etl.process_and_predict_housing_price import ProcessPredictHousingPrice

MODULE_NAME = "etl.process_and_predict_housing_price"
HOUSING_DATA_PATH = Path(__file__).parent.parent.joinpath("data", "housing.csv")

predict_price_df = pd.DataFrame(
    {
        "longitude": [-122.64, -115.73, -117.96],
        "latitude": [38.01, 33.35, 33.89],
        "housing_median_age": [36.0, 23.0, 24.0],
        "total_rooms": [1336.0, 1586.0, 1332.0],
        "total_bedrooms": [258.0, 448.0, 252.0],
        "population": [678.0, 338.0, 625.0],
        "households": [249.0, 182.0, 230.0],
        "median_income": [5.5789, 1.2132, 4.4375],
        "ocean_proximity": ["NEAR OCEAN", "INLAND", "<1H OCEAN"],
    }
)

non_standardized_columns_df = pd.DataFrame(
    {
        "LONGITUDE": [-122.64, -115.73, -117.96],
        "LAT": [38.01, 33.35, 33.89],
        "MEDIAN_AGE": [36.0, 23.0, 24.0],
        "ROOMS": [1336.0, 1586.0, 1332.0],
        "BEDROOMS": [258.0, 448.0, 252.0],
        "POP": [678.0, 338.0, 625.0],
        "HOUSEHOLDS": [249.0, 182.0, 230.0],
        "MEDIAN_INCOME": [5.5789, 1.2132, 4.4375],
        "OCEAN_PROXIMITY": ["NEAR OCEAN", "INLAND", "<1H OCEAN"],
    }
)

model_ready_df = pd.DataFrame(
    {
        "longitude": [-122.64, -115.73, -117.96],
        "latitude": [38.01, 33.35, 33.89],
        "housing_median_age": [36.0, 23.0, 24.0],
        "total_rooms": [1336.0, 1586.0, 1332.0],
        "total_bedrooms": [258.0, 448.0, 252.0],
        "population": [678.0, 338.0, 625.0],
        "households": [249.0, 182.0, 230.0],
        "median_income": [5.5789, 1.2132, 4.4375],
        "ocean_proximity_NEAR OCEAN": [1.0, 0.0, 0.0],
        "ocean_proximity_INLAND": [0.0, 1.0, 0.0],
        "ocean_proximity_<1H OCEAN": [0.0, 0.0, 1.0],
        "ocean_proximity_NEAR BAY": [0.0, 0.0, 0.0],
        "ocean_proximity_ISLAND": [0.0, 0.0, 0.0],
    }
)


@pytest.fixture
def mock_save_to_db(mocker):
    return mocker.patch(f"{MODULE_NAME}.ProcessPredictHousingPrice.save_to_db")


def test_preprocess_data():
    # Prepare
    predictor = ProcessPredictHousingPrice(input_df=predict_price_df)
    # Act
    predictor.preprocess_data()
    # Assert
    assert_frame_equal(
        predictor.data_frame.sort_index(axis=1), model_ready_df.sort_index(axis=1)
    )


def test_preprocess_data_column_names():
    # Prepare
    predictor = ProcessPredictHousingPrice(input_df=non_standardized_columns_df)
    # Act
    predictor.preprocess_data()
    # Assert
    assert_frame_equal(
        predictor.data_frame.sort_index(axis=1), model_ready_df.sort_index(axis=1)
    )


def test_predict_housing_price():
    # Prepare
    predictor = ProcessPredictHousingPrice(input_df=predict_price_df)
    # Act
    predicted_values = predictor.predict_housing_price()
    # Assert
    assert predicted_values == [
        320201.58554043656,
        58815.45033764739,
        192575.77355634805,
    ]


def test_evaluate_and_save_with_housing_data(mock_save_to_db):
    # Prepare
    predictor = ProcessPredictHousingPrice(input_data_path=HOUSING_DATA_PATH)
    # Act
    predictor.evaluate_and_save()
    # Assert
    assert mock_save_to_db.call_count == 2


def test_prepare_test_train_data_with_predict_column_absent():
    # Prepare
    housing_df = pd.read_csv(HOUSING_DATA_PATH)
    housing_df.drop(columns=["MEDIAN_HOUSE_VALUE"], inplace=True)
    predictor = ProcessPredictHousingPrice(input_df=housing_df)
    # Act and assert
    with pytest.raises(PredictColumnNotFoundException):
        predictor.prepare_data_test_train()


def test_predict_housing_price_with_categorical_column_absent():
    # Prepare
    predict_price_df.drop(columns=CATEGORICAL_COLUMN, inplace=True)
    predictor = ProcessPredictHousingPrice(input_df=predict_price_df)
    # Act and assert
    with pytest.raises(CategoricalColumnNotFoundException):
        predictor.predict_housing_price()


def test_no_data_to_process_provided():
    # Prepare, act, assert
    with pytest.raises(InvalidInputException):
        predictor = ProcessPredictHousingPrice()
