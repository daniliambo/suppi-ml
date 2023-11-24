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


def plot(df):
    min_price = df['min_price']
    max_price = df['max_price']
    avg_price = df['marketing_seller_price']
    avg_price = np.random.choice(avg_price, size=len(min_price), replace=True)
    def f(price, clr): return plt.hist(price, label=str(price), bins=100,
                                       color=clr, log=False, range=[0, 2000], stacked=False, alpha=0.3)

    f(min_price, 'r')
    f(max_price, 'g')
    f(avg_price, 'b')

    print(
        """min_price, 'r'
max_price, 'g'
avg_price, 'b'"""
    )
    plt.show()


def preprocess_meta(df):
    df.drop(['min_profitability', 'max_profitability',
            'Unnamed: 0'], axis=1, inplace=True)
    return df


def add_missing_offers(df_feature: pd.DataFrame, range: int = 0.3):
    column_to_group_by = 'offer_id'
    column_to_aggregate_on = 'marketing_seller_price'
    prices = df_feature.groupby([column_to_group_by])[
        column_to_aggregate_on].agg('mean')
    return prices


def fill_missing_min_max(df_meta, prices, n_feature):
    prices['shop_id'] = pd.Series(
        ['479580b8-30b1-4ace-9da1-77649e3c39ee'] * n_feature)

    columns_to_merge_on = ['shop_id', 'offer_id']
    df = pd.merge(prices, df_meta,  how='left', left_on=columns_to_merge_on,
                  right_on=columns_to_merge_on)
    print(df.columns)

    df['min_price'].fillna(df['marketing_seller_price'] - np.std(
        df['marketing_seller_price'] - df['min_price']), inplace=True)

    df['max_price'].fillna(df['marketing_seller_price'] + np.std(
        df['marketing_seller_price'] - df['max_price']), inplace=True)

    return df


def price_bundles(X, n) -> pd.DataFrame:
    return pd.concat(X.apply(lambda x: pd.DataFrame({"shop_id": [x['shop_id']] * n,
                                                     "offer_id": [x['offer_id']] * n,
                                                     "bundle": np.linspace(x['min_price'], x['max_price'], num=n).tolist()}), axis=1).tolist())


def compose_pipeline():
    reader = ReadData()
    df_meta = reader.read('meta')
    df_feature = reader.read('feature')

    df_meta = preprocess_meta(df_meta)
    prices = add_missing_offers(df_feature).reset_index()
    df = fill_missing_min_max(df_meta, prices, df_feature.shape[0])
    return df


def run_pipeline(n):
    df = compose_pipeline()
    bundles = price_bundles(df, n)
    return df, bundles


def load_and_preprocess():
    n = 10

    df, bundles = run_pipeline(n)
    return bundles


if __name__ == "__main__":
    bundles = load_and_preprocess()
