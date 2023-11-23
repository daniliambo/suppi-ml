from utils.reader import ReadData
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


def compose_pipeline():

    def compose_encode_decode_dict(columns_feature):
        d_encode = dict(zip(columns_feature, np.arange(
            len(columns_feature), dtype=int)))

        d_decode = dict(zip(np.arange(len(columns_feature),
                        dtype=int), columns_feature))

        def h_encode(x): return list(map(lambda y: d_encode[y], x))
        def h_decode(x): return list(map(lambda y: d_decode[y], x))
        return d_encode, d_decode, h_encode, h_decode

    columns_base = ["date", "offer_id", "shop_id"]

    columns_main = ['num_actions',
                    'position_category',
                    'price_index',
                    'external_index_data_minimal_price',
                    'external_index_data_price_index_value',
                    'self_marketplaces_index_data_minimal_price',
                    'self_marketplaces_index_data_price_index_value',
                    'marketing_seller_price',
                    'marketing_price']

    conversions = ['conv_tocart',
                   'conv_tocart_pdp',
                   'hits_tocart',
                   'hits_tocart_pdp',
                   'hits_view_search']

    columns_feature = columns_base + columns_main + conversions

    d_encode, d_decode, h_encode, h_decode = compose_encode_decode_dict(
        columns_feature)

    keep_feature = Pipeline([
        ('featureUnion', FeatureUnion([
            ("columns_to_keep_feature", ColumnTransformer(
                [('columns_to_keep_feature1', 'passthrough', columns_base + columns_main)], remainder='drop')),
            ("add_conversions", ColumnTransformer(
                [('columns_to_add', SimpleImputer(strategy='constant', fill_value=0, add_indicator=False, keep_empty_features=True), conversions)], remainder='drop'))
        ]))
    ])

    f = set

    columns_to_standardize_feature = list(
        f(columns_feature) - f(columns_base))

    ordinal_encode_base_columns_feature = Pipeline([
        ("ordinal_encode_base_columns_feature", ColumnTransformer([("ordinal",
                                                                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=None, max_categories=None
                                                                                   ), h_encode(columns_base))], remainder='drop', sparse_threshold=0, n_jobs=-1)),
    ])

    standardize_numeric_columns_feature = Pipeline(steps=[
        ("standardize_feature", ColumnTransformer(
            [('standardize_feature', StandardScaler(), h_encode(columns_to_standardize_feature))])),
    ])

    impute_na_columns_feature = Pipeline([
        ("KNN_imputer_feature", ColumnTransformer(
            [('KNN_imputer_feature', KNNImputer(add_indicator=False, keep_empty_features=True), np.arange(len(columns_feature), dtype=int))], verbose_feature_names_out=True))
    ])

    unite_columns_feature = Pipeline([("FeatureUnion",
                                       FeatureUnion(
                                           transformer_list=[('ordinal_encode_base_columns_feature', ordinal_encode_base_columns_feature),
                                                             ('standardize_numeric_columns_feature', standardize_numeric_columns_feature)]
                                       ))])

    combine_knn_imputation_feature = Pipeline(steps=[
        ('unite_columns_feature', unite_columns_feature),
        ('impute_na_columns_feature', impute_na_columns_feature),
    ])

    preprocessor_before_join_feature = Pipeline(steps=[
        ('keep_feature', keep_feature),
        ('knn', combine_knn_imputation_feature),
    ], verbose=True)

    class DataHolder:
        def __init__(self, d_encode=None, d_decode=None, h_encode=None, h_decode=None, columns_base=None, columns_main=None, conversions=None, preprocessor_before_join_feature=None, ordinal_encode_base_columns_feature=None) -> None:
            self.d_encode = d_encode
            self.d_decode = d_decode
            self.h_encode = h_encode
            self.h_decode = h_decode
            self.columns_base = columns_base
            self.columns_main = columns_main
            self.conversions = conversions
            self.preprocessor_before_join_feature = preprocessor_before_join_feature
            self.ordinal_encode_base_columns_feature = ordinal_encode_base_columns_feature

    dh = DataHolder()
    dh.d_encode = d_encode
    dh.d_decode = d_decode
    dh.h_encode = h_encode
    dh.h_decode = h_decode
    dh.columns_base = columns_base
    dh.columns_main = columns_main
    dh.conversions = conversions
    dh.preprocessor_before_join_feature = preprocessor_before_join_feature
    dh.ordinal_encode_base_columns_feature = ordinal_encode_base_columns_feature

    return dh


def run_pipeline(dh, df):
    X = dh.preprocessor_before_join_feature.fit_transform(df)
    res = pd.DataFrame(X)

    # rewrite
    inv_cols = dh.ordinal_encode_base_columns_feature['ordinal_encode_base_columns_feature'].transformers_[
        0][1].inverse_transform(res.loc[:, dh.h_encode(dh.columns_base)])
    res.loc[:, dh.h_encode(dh.columns_base)] = inv_cols

    res.rename(columns=dh.d_decode, inplace=True)
    return res


def load_and_preprocess():
    dh = compose_pipeline()
    reader = ReadData()
    df = reader.read('feature')
    df = run_pipeline(dh, df)
    return df
