import logging
from pathlib import Path
from typing import List

import joblib
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from etl.constants import (
    CATEGORICAL_COLUMN,
    CATEGORICAL_VALUES,
    COLUMN_MAPPING,
    MODEL_PATH,
    NA_VALUES,
    PREDICTED_DATA_FILENAME,
    PREDICTED_DATA_FILEPATH,
    PREDICT_COLUMN,
    RANDOM_STATE,
    TRANSFORMED_DF_FILENAME,
    TRANSFORMED_DF_FILEPATH,
)

from etl.exceptions import (
    CategoricalColumnNotFoundException,
    CategoricalDataUnrecognizedException,
    InvalidInputException,
    PredictColumnNotFoundException,
)

from sqlalchemy import create_engine

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger()


class ProcessPredictHousingPrice:
    """Class to preprocess, predict house prices and save information to database"""

    def __init__(self, input_data_path=None, input_df=pd.DataFrame()):
        self.input_data_path = input_data_path
        self.input_df = input_df
        if not self.input_data_path and self.input_df.empty:
            msg = "Either file path or test dataframe should be provided."
            raise InvalidInputException(msg)
        if self.input_data_path and not self.input_df.empty:
            msg = "Either file path or test dataframe should be provided."
            raise InvalidInputException(msg)
        if self.input_data_path:
            self.data_frame = pd.read_csv(self.input_data_path, na_values=NA_VALUES)
        elif not self.input_df.empty:
            self.data_frame = input_df

    @staticmethod
    def predict(X: pd.DataFrame, model: RandomForestRegressor) -> numpy.ndarray:
        """Predicts the housing price

        Parameters
        ----------
        X: pd.DataFrame
            Feature dataframe from which house price should be predicted
        model: RandomForestRegressor
            Trained model

        Returns
        -------
        numpy.ndarray:
            Array containing Housing price predictions
        """
        Y = model.predict(X)
        return Y

    @staticmethod
    def load_model():
        """Loads prediction model from given path"""
        model = joblib.load(MODEL_PATH)
        return model

    @staticmethod
    def save_to_db(data: pd.DataFrame, filepath: Path, filename: str):
        """Saves the dataframe to sqlite database

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe to save to database
        filepath: Path
            Save location path
        filename: str
            Filename
        """
        engine = create_engine(f"sqlite:///{filepath}")
        if isinstance(data, pd.DataFrame):
            data.to_sql(
                filename.split(".")[0], engine, if_exists="replace", index=False
            )
            logger.info(f"{filename=} saved to database")
        else:
            logger.error("Please provide valid dataframe to save to db")

    def _prepare_dataframe_for_model(self):
        """Prepares the dataframe to fit the model. Makes sure all the required columns are present and
        adds the required columns if absent with value as 0
        """
        df = pd.DataFrame(columns=self._required_column_names())
        self.data_frame = pd.concat([df, self.data_frame])
        self.data_frame.fillna(value=0, axis=1, inplace=True)

    def _validate_categorical_data(self):
        """Validates categorical data in the dataframe

        Raises
        ------
        CategoricalColumnNotFoundException
            If categorical column not present in dataframe
        CategoricalDataUnrecognizedException
            If unrecognized categorical data present in the column
        """
        try:
            odd_data = ", ".join(
                list(set(self.data_frame[CATEGORICAL_COLUMN]) - set(CATEGORICAL_VALUES))
            )
        except KeyError:
            raise CategoricalColumnNotFoundException(f"{CATEGORICAL_COLUMN=} absent")
        if odd_data:
            raise CategoricalDataUnrecognizedException(
                f"Unrecognized data '{odd_data}' present in {CATEGORICAL_COLUMN=}"
            )

    def _required_column_names(self) -> List:
        """Loads the model and returns required columns by the model

        Returns
        -------
        List
            List of columns expected by the model
        """
        model = self.load_model()
        column_names = model.feature_names_in_.tolist()
        return column_names

    def _cleanse_data(self):
        """Cleanses the given data as part of preprocessing. Performs below operations:
        1.  Rename the column names as expected by the model
        2.  Encode categorical column
        3.  Drops the columns present in dataframe which are not required by the model
        """
        logger.info("Cleansing the data...")
        self.data_frame.rename(columns=COLUMN_MAPPING, inplace=True)
        self.data_frame.dropna(inplace=True)

        try:
            self._validate_categorical_data()
        except CategoricalDataUnrecognizedException as e:
            logger.warning(str(e.message))

        # encode the categorical variables
        self.data_frame = pd.get_dummies(self.data_frame, columns=[CATEGORICAL_COLUMN])

        # Drop unexpected columns
        self.data_frame.drop(
            columns=[
                col
                for col in self.data_frame.columns
                if col not in self._required_column_names() + PREDICT_COLUMN
            ],
            axis=1,
            inplace=True,
        )

    def preprocess_data(self):
        """Pre-process data to make it ready to be accepted by the model"""

        self._cleanse_data()
        self._prepare_dataframe_for_model()

    def prepare_data_test_train(self):
        """Segregates features and column to be predicted. Splits training and test data

        Raises
        ------
        PredictColumnNotFoundException
            If column to predict is absent in dataframe
        """

        self.preprocess_data()
        try:
            df_features = self.data_frame.drop(columns=PREDICT_COLUMN, axis=1)
        except KeyError:
            raise PredictColumnNotFoundException("Column to predict absent")
        y = self.data_frame[PREDICT_COLUMN].values
        X_train, X_test, y_train, y_test = train_test_split(
            df_features, y, test_size=0.2, random_state=RANDOM_STATE
        )
        return (X_train, X_test, y_train, y_test)

    def evaluate_and_save(self):
        """
        Saves preprocessed to database. Evaluates the model using preprocessed data and save the predicted value
        to database
        """

        logger.info("Preparing the data...")
        X_train, X_test, y_train, y_test = self.prepare_data_test_train()
        self.save_to_db(
            data=self.data_frame,
            filepath=TRANSFORMED_DF_FILEPATH,
            filename=TRANSFORMED_DF_FILENAME,
        )
        logger.info("Loading the model...")
        model = self.load_model()

        logger.info("Calculating train dataset predictions...")
        y_pred_train = self.predict(X_train, model)

        logger.info("Calculating test dataset predictions...")
        y_pred_test = self.predict(X_test, model)

        # evaluate model
        logger.info("Evaluating the model...")
        train_error = mean_absolute_error(y_train, y_pred_train)
        test_error = mean_absolute_error(y_test, y_pred_test)

        logger.info("First 5 predictions:")
        logger.info(f"\n{X_test.head()}")
        logger.info(y_pred_test[:5])
        logger.info(f"Train error: {train_error}")
        logger.info(f"Test error: {test_error}")

        logger.info("Saving prediction to database...")
        predicted_df = pd.DataFrame(
            [{"Predicted_train_data": y_pred_train, "Predicted_test_data": y_pred_test}]
        )
        self.save_to_db(
            data=predicted_df,
            filepath=PREDICTED_DATA_FILEPATH,
            filename=PREDICTED_DATA_FILENAME,
        )

    def predict_housing_price(self):
        """Preprocess given data, load the model and predicts the housing price"""
        self.preprocess_data()
        logger.info("Loading the model...")
        model = self.load_model()
        y_pred = self.predict(self.data_frame, model)
        logger.info(f"Predicted price is : {y_pred}")
        return list(y_pred)
