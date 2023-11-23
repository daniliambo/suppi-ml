from utils.reader import ReadData
import warnings
from pipeline.step_0.custom_transformers import ColumnTransformerDrop, ColumnTransformerDropNaBase, ColumnTransformerDropNonInformative, ColumnTransformerFillNaData, ColumnTransformerGroupBy, ColumnTransformerJoinNonInformative, ColumnTransformerOrdinalEncoding
import os
from collections import defaultdict
import pprint


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

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

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pprint.PrettyPrinter(indent=4)
warnings.simplefilter(action='ignore', category=FutureWarning)


def compose_pipeline():
    # step_1
    columns_to_drop_data = ['id', 'sku']

    drop_data = Pipeline([
        ("drop_data", ColumnTransformerDrop(columns=columns_to_drop_data))
    ])

    # step_2
    columns_to_drop_na_base_data = ['shop_id', 'offer_id', 'date']

    drop_na_base_data = Pipeline([
        ("drop_na_base_data", ColumnTransformerDropNaBase(
            columns=columns_to_drop_na_base_data))
    ])

    # step_3
    columns_to_drop_fill_na_data = ['warehouse_name', 'delivery_schema',
                                    'gds_profitability_after_tax', 'shop_profitability_after_tax']

    drop_fill_na_data = Pipeline([
        ("drop_fill_na_data", ColumnTransformerFillNaData(
            columns=columns_to_drop_fill_na_data))
    ])

    # step_4
    drop_non_informative_data = Pipeline([
        ("drop_non_informative_data", ColumnTransformerDropNonInformative())
    ])

    # step_5
    columns_to_join_na_data = [
        'premium_service_cost',
        'cross_docking_cost',
        'placement_cost',
        'compensations',
        'direct_logistics',
        'installment_sale_cost',
        'last_mile',
        'acquiring_cost_prod',
        'orders_commission',
        'returns_commission',
        'reverse_logistics',
        'other'
    ]

    join_non_informative_data = Pipeline(steps=[
        ("join_non_informative_data", ColumnTransformerJoinNonInformative(
            columns=columns_to_join_na_data)),
    ])

    # step_6

    columns_to_group_by_day_data = ['shop_id', 'offer_id', 'date']

    group_by_day_data = Pipeline(steps=[
        ("group_by_day_data", ColumnTransformerGroupBy(
            group_by_columns=columns_to_group_by_day_data)),
    ])

    # step_7
    columns_to_oridal_encode_data = ['warehouse_name', 'delivery_schema']
    oridal_encode_data = Pipeline(steps=[
        ("oridal_encode_data", ColumnTransformerOrdinalEncoding(
            columns_to_oridal_encode_data)),
    ])

    preprocessor_before_join_data = Pipeline(steps=[
        ('drop_data', drop_data),
        ('drop_na_base_data', drop_na_base_data),
        ('drop_fill_na_data', drop_fill_na_data),
        ("drop_non_informative_data", drop_non_informative_data),
        ("join_non_informative_data", join_non_informative_data),
        ("group_by_day_data", group_by_day_data),
        # ("oridal_encode_data", oridal_encode_data),
    ], verbose=True)
    return preprocessor_before_join_data


def run_pipeline(preprocessor_before_join_data, df):
    X = preprocessor_before_join_data.fit_transform(df)
    return X


def load_and_preprocess():
    reader = ReadData()
    df_data = reader.read('data')
    preprocessor_before_join_data = compose_pipeline()
    df = run_pipeline(preprocessor_before_join_data, df_data)
    return df


if __name__ == "__main__":
    load_and_preprocess()
