import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# This function creates a dataframe that shows how many missing rows and what % of the total number of rows there are for each column in a dataframe.
def missing_rows(df):
    # The missing row % is = to the sum of all the nulls divided by the total number of rows
    missing_row_pct = (df.isnull().sum() / len(df)) * 100
    # This is the total number of missing rows per column
    missing_row_raw = df.isnull().sum()
    # creating a new dataframe to return that contains all the information we are looking for
    missing_df = pd.DataFrame({'num_rows_missing': missing_row_raw, 'pct_rows_missing': missing_row_pct})
    return missing_df

# This function creates dummy variables for the room_type column
def airbnb_dummies(df):
    # Create dummy variables for the room_type column
    type_dummies = pd.get_dummies(df['room_type'], drop_first=False)
    # rename these dummy columns to each of the room types
    type_dummies = type_dummies.rename(columns={type_dummies.columns[0] : 'entire_home', type_dummies.columns[1] : 'hotel_room', type_dummies.columns[2] : 'private_room', type_dummies.columns[3] : 'shared_room'})
    # attach these new dummy columns to the original dataframe
    df = pd.concat([df, type_dummies], axis=1)
    return df

# This function splits a dataframe into train, validate, and test dataframes.
def df_split(df):
    # Creating two data frames, a larger one with train and validate combined, and the test dataframe
    train_validate, test = train_test_split(df, test_size=.25, random_state=123)
    # Splitting the train_validate dataframe in to separate dataframes for each.
    train, validate = train_test_split(train_validate, test_size=.4, random_state=123)
    return train, validate, test

# This function creates scaled copies of the train, validate, and test dataframes.
def airbnb_scaler(train, validate, test):
    # Creating copies of the data frames so that we don't modify the originals
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # Creating MinMaxScaler
    scaler = MinMaxScaler()
    # Grabbing the columns for each dataframe that we want to scale
    train_to_scale = train_scaled[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']]
    validate_to_scale = validate_scaled[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']]
    test_to_scale = test_scaled[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']]
    # Scaling the columns for each dataframe
    train_scaled[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']] = scaler.fit_transform(train_to_scale)
    validate_scaled[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']] = scaler.fit_transform(validate_to_scale)
    test_scaled[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']] = scaler.fit_transform(test_to_scale)
    return train_scaled, validate_scaled, test_scaled

# This function combines a few of the preparation functions together in order to quickly prep data for exploration.
def prep_data(df):
    df = df.drop(columns='neighbourhood_group')
    df = airbnb_dummies(df)
    train, validate, test = df_split(df)
    train_scaled, validate_scaled, test_scaled = airbnb_scaler(train, validate, test)
    return train, validate, test, train_scaled, validate_scaled, test_scaled