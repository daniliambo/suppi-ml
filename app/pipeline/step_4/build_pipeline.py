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


def visualization(df):
    df = set_datetime(df)
    print("time_period: from {} to {}".format(df.index[0], df.index[-1]))
    plot_distributions(df)
