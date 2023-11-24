from pipeline.step_2 import build_pipeline as build_pipeline_meta
from models import model as model_py
import imp
from pandas.tseries.offsets import DateOffset
from pipeline.step_3.build_pipeline import load_and_preprocess as load_and_preprocess_union

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import pprint
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import catboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.special import softmax

from tqdm import tqdm
from warnings import filterwarnings
from IPython.display import display

filterwarnings("ignore")
pd.set_option('display.max_columns', None)
np.random.seed(42)


np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pprint.PrettyPrinter(indent=4)


def set_datetime(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date', drop=True, inplace=False)
    df.sort_index(inplace=True)
    return df


def plot_distributions(df):

    def plot_sales_hist(col_to_group_by, col, df):
        val = df.groupby([col_to_group_by])[
            col].sum().values

        plt.figure(figsize=(16, 6))
        plt.grid()
        plt.title('quantity_of_paid_with_in_month')
        plt.hist(val, label=col, bins=100,
                 color='g', log=True, range=[0, 300], stacked=False, density=True, alpha=0.3)
        plt.legend()
        plt.show()

    def plot_sales_line(col_to_group_by, col, df):
        val = df.groupby([col_to_group_by])[col].sum().reset_index()
        val.sort_values(by=[col], ascending=False, inplace=True)

        offer_ids = val.iloc[:5, 0]
        plt.figure(figsize=(16, 6))
        plt.grid()
        plt.title('quantity_of_paid_by_day')

        for offer in offer_ids:
            series = df.loc[df['offer_id'] == offer, 'quantity_of_paid']
            plt.plot(series)

        plt.legend()
        plt.show()

    TARGET = "quantity_of_paid"
    col_to_group_by = 'offer_id'

    plot_sales_hist(col_to_group_by, TARGET, df)
    plot_sales_line(col_to_group_by, TARGET, df)


# def train_test_split(df):

#     def get_split(df: pd.DataFrame):
#         offset = df.index[-1] - DateOffset(weeks=1)
#         df_train = df[df.index < offset]
#         df_test = df[df.index >= offset]
#         df_train.reset_index(inplace=True)
#         df_test.reset_index(inplace=True)
#         return df_train, df_test

#     df = set_datetime(df)
#     df_train, df_test = get_split(df)

#     return df_train, df_test


def set_target(df):
    TARGET = 'TARGET'
    qp, qr = 'quantity_of_paid', 'quantity_of_return',
    df[TARGET] = df[qp] - df[qr]
    df.drop([qp, qr], axis=1, inplace=True)
    return df


def visualization(df):
    df = set_datetime(df)
    print("time_period: from {} to {}".format(df.index[0], df.index[-1]))
    plot_distributions(df)


def compose_pipeline():
    df = load_and_preprocess_union()
    df = set_target(df)
    df = set_datetime(df)
    return df


def make_predictions(df):
    # TRIM DATES
    df = df[df.index <= df.index[-1] - pd.Timedelta(16, "d")]

    load_and_preprocess_meta = build_pipeline_meta.load_and_preprocess
    df_meta = load_and_preprocess_meta()

    model = model_py.Model()
    shop_id = df.shop_id[0]
    offer_ids = df.groupby('offer_id')['TARGET'].sum(
    ).sort_values(ascending=False)[:10].index

    counter = 0
    for offer_id in offer_ids:
        if counter == 4:
            break
        counter += 1

        df_run = df.loc[(df.shop_id == shop_id) & (df.offer_id == offer_id)]
        df_run.drop(['shop_id', 'offer_id', 'title', 'warehouse_name',
                    'delivery_schema'], axis=1, inplace=True)

        price = df_run['marketing_seller_price']
        cols_to_drop = set(df.columns[df[df_run > 0].sum() == 0]) - set(['shop_id', 'offer_id', 'title', 'warehouse_name',
                                                                        'delivery_schema'])

        df_run.drop(cols_to_drop, axis=1, inplace=True)
        df_run['marketing_seller_price'] = price

        model = model_py.Model()
        model.main(inference=False, df=df_run)

        df_meta_run = df_meta.loc[(df_meta.shop_id == shop_id) & (
            df_meta.offer_id == offer_id)]

        prices = [df_meta_run.index[0], df_meta_run.index[-1]]
        for price in prices:
            df_run['marketing_seller_price'] = [price] * df_run.shape[0]
            model.main(inference=True, df=df_run)
