# Directories

## 1. data

### Files

    a. housing.csv : csv file to evaluate the model
    b. predictions.db : Test and train predictions data in sqlite database
    c. transformed.db : Transformed housing.csv file to fit the model

## 2. etl

### Modules

    a. constants.py : Stores all the constants in use
    b. exceptions.py : Contains custom exceptions for expected failures
    c. model.joblib : Already trained model
    d. process_and_predict_housing_price.py : Contains ProcessPredictHousingPrice class which has methods to preprocess data, save data to database and predict housing price

## 3. tests

### Modules

    a. test_process_and_predict_housing_price.py : Unit test cases to make sure the methods to preprocess, save data and predictions are working as expected

# How to evaluate the model and predict the housing price?

### 1. How to preprocess the training data, evaluate the model, save transformed file and predictions to detabase

##### ProcessPredictHousingPrice class accepts argument`input_data_path`:

`predictor = ProcessPredictHousingPrice(input_data_path=HOUSING_DATA_PATH)`<br/>
`predictor.evaluate_and_save()`<br/>

### 2. How to preprocess any new feature dataframe and predict housing price

##### ProcessPredictHousingPrice class accepts argument`input_df`:

`predictor = ProcessPredictHousingPrice(input_df=predict_price_df)`<br/>
`predicted_values = predictor.predict_housing_price()`   