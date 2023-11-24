
import os
from collections import defaultdict
import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import pipeline
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

import os
from collections import defaultdict
import pprint

from utils.reader import ReadData
from pipeline.step_0.build_pipeline import load_and_preprocess as load_and_preprocess_data
from pipeline.step_1.build_pipeline import load_and_preprocess as load_and_preprocess_feature
from pipeline.step_3.custom_transformers import ColumnTransformerSetDate


def compose_pipeline():

    def compose_encode_decode_dict(columns_feature):
        d_encode = dict(zip(columns_feature, np.arange(
            len(columns_feature), dtype=int)))

        d_decode = dict(zip(np.arange(len(columns_feature),
                        dtype=int), columns_feature))

        def h_encode(x): return list(map(lambda y: d_encode[y], x))
        def h_decode(x): return list(map(lambda y: d_decode[y], x))
        return d_encode, d_decode, h_encode, h_decode

    def load_data():
        data_loader = load_and_preprocess_data
        feature_loader = load_and_preprocess_feature

        df_data = data_loader()
        df_feature = feature_loader()

        columns_base = ["date", "offer_id", "shop_id"]

        df = pd.merge(df_data, df_feature,  how='left', left_on=columns_base,
                      right_on=columns_base)
        return df

    df = load_data()

    d_encode, d_decode, h_encode, h_decode = compose_encode_decode_dict(
        df.columns)

    impute_features = Pipeline([
        ("simple_imputer", ColumnTransformer(
            [('simple_imputer', SimpleImputer(strategy='constant', fill_value=0, add_indicator=False, keep_empty_features=True), h_encode(df.columns[df.isnull().any()]))], remainder='passthrough', verbose_feature_names_out=False))
    ])

    set_datetime = Pipeline([
        ("set_datetime", ColumnTransformerSetDate(
            d_encode, d_decode, h_encode, h_decode))
    ])

    preprocessor_data = Pipeline(steps=[
        ("simple_imputer", impute_features),
        # ('set_datetime', set_datetime),
    ])

    df = preprocessor_data.fit_transform(df)
    df = pd.DataFrame(df, columns=preprocessor_data.get_feature_names_out())
    return df, preprocessor_data


def run_pipeline():
    df, preprocessor_data = compose_pipeline()
    return df, preprocessor_data


def load_and_preprocess():
    df, preprocessor_data = run_pipeline()
    return df
