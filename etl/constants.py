from pathlib import Path

NA_VALUES = ["Null"]
COLUMN_MAPPING = {
    "LONGITUDE": "longitude",
    "LAT": "latitude",
    "MEDIAN_AGE": "housing_median_age",
    "MEDIAN_HOUSE_VALUE": "median_house_value",
    "OCEAN_PROXIMITY": "ocean_proximity",
    "ROOMS": "total_rooms",
    "BEDROOMS": "total_bedrooms",
    "POP": "population",
    "HOUSEHOLDS": "households",
    "MEDIAN_INCOME": "median_income",
}
CATEGORICAL_COLUMN = "ocean_proximity"
CATEGORICAL_VALUES = ["NEAR OCEAN", "INLAND", "<1H OCEAN", "ISLAND", "NEAR BAY"]
PREDICT_COLUMN = ["median_house_value"]
MODEL_PATH = Path(__file__).parent.joinpath("model.joblib")
RANDOM_STATE = 100
TRANSFORMED_DF_FILENAME = "transformed.db"
PREDICTED_DATA_FILENAME = "predictions.db"
TRANSFORMED_DF_FILEPATH = Path(__file__).parent.parent.joinpath("data", TRANSFORMED_DF_FILENAME)
PREDICTED_DATA_FILEPATH = Path(__file__).parent.parent.joinpath("data", PREDICTED_DATA_FILENAME)
