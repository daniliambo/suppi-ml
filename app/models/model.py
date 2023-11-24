import pandas as pd
import numpy as np
import catboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.special import softmax
from typing import List

from tqdm import tqdm
from warnings import filterwarnings
from IPython.display import display

filterwarnings("ignore")
pd.set_option('display.max_columns', None)
np.random.seed(42)


class ModelBuilderUtils(object):
    def __init__(self):
        self.metrics = {"MAE": [], "RMSE": [],
                        "MAPE": [], "R2": []}  # init metrics dict

    def preprocess_df(self, df):
        self.columns_to_nums = dict(
            zip(df.columns, np.arange(len(df.columns), dtype=int)))

        self.nums_to_columns = dict(
            zip(np.arange(len(df.columns), dtype=int), df.columns))

        # rename cols
        df.rename(columns=self.columns_to_nums, inplace=True)

        # rename "0" back to target
        df.rename(
            columns={self.columns_to_nums['TARGET']: "TARGET"}, inplace=True)

        return df

    def split_df(self, df):
        # split the data
        df_train = df[df.index <= df.index[-1] - pd.Timedelta(7, "d")]
        df_test = df[df.index > df.index[-1] - pd.Timedelta(7, "d")]
        return df_train, df_test

    def score_model(self, y_trues, y_preds):
        # create scores
        self.metrics["MAE"].append(
            round(mean_absolute_error(y_trues, y_preds), 2))
        self.metrics["MAPE"].append(
            mean_absolute_percentage_error(y_trues, y_preds))
        self.metrics["RMSE"].append(
            round(mean_squared_error(y_trues, y_preds, squared=False), 2))
        self.metrics["R2"].append(r2_score(y_trues, y_preds))

    def display_predictions(self, timeseries, predict):
        # display predictions
        plt.figure(dpi=150)
        plt.figure(figsize=(16, 6))
        plt.grid()
        plt.plot(timeseries, color="blue", label="true")
        plt.plot(predict, color="green", label="predictions")
        plt.legend(loc="upper left")
        plt.show()

    def feature_importance(self, model, path="./output/feature_importance.csv"):
        fi = softmax(model.get_feature_importance()[3:])
        cols = list(self.nums_to_columns.values())[1:]
        pd.DataFrame([fi], columns=cols, index=np.arange(
            len(cols), dtype=int)).iloc[0, :].to_csv(path)

    # to implement
    def save_model(self, path: str = "./model.pth"):
        pass

    # to implement
    def load_model(self, path: str = "./model.pth"):
        pass


class ModelBuilderFeatureExtractor:
    def __init__(self):
        self.window = None  # window for MA and strides; instantiated in ModelBuilder
        self.weekly_mean = None  # df with weekly_mean

    def enreach_data_1(self, df):
        # append weekly_mean and weekly_sin to df
        df['weekly_mean'] = df.groupby(df.index.weekday)[
            'TARGET'].transform(np.mean)
        df['weekly_sin'] = list(
            map(lambda x: np.sin(x / 3.82), df.index.weekday))
        return df

    def enreach_data_2(self, df_train, window: int = 7):
        # calculate rolling mean
        df_rolling = df_train.rolling(window).mean()  # removed closed="left"

        # set names for new dfs
        df_rolling.columns = list(
            map(lambda x: "rolling_mean_" + str(x), df_train.columns))

        print(self.weekly_mean.shape)
        print(df_rolling.shape)
        # fill in the missing
        df_rolling.to_numpy()[:window] = self.weekly_mean.to_numpy()[:window]

        # drop rolling_mean_TARGET
        df_rolling = df_rolling.drop(
            ['rolling_mean_TARGET'], axis=1, inplace=False)

        # concat with the df_train
        return pd.merge(df_train, df_rolling, left_index=True, right_index=True)

    def extract_weekly_mean(self, df_train):
        # calculate weekly by week; USE IT ON TRAIN
        tmp = pd.DataFrame.copy(df_train)
        tmp['week'] = tmp.index.day_of_week
        preds_by_week = tmp.groupby(['week']).mean()

        # stores result in var
        self.weekly_mean = pd.DataFrame.copy(preds_by_week)

    def select_period(self, df_train, model_idx):
        # selects period for transformation
        return df_train[-(self.window + model_idx):].values

    def extract_window(self, X, idx):

        # don't forget to [:-1]
        X = np.apply_along_axis(
            lambda x: np.lib.stride_tricks.sliding_window_view(x, window_shape=(idx + self.window)), arr=X,
            axis=0)[:-1]

        X = np.reshape(X, newshape=(
            X.shape[0], X.shape[1] * X.shape[2]), order='F')
        return X

    def generate_new_df(self, df_test, predict, idx):
        # generates new df for predict
        index = df_test.index[idx]
        df = df_test.loc[[index]]
        df['TARGET'] = [predict]
        return df


class ModelBuilder(ModelBuilderUtils, ModelBuilderFeatureExtractor):
    def __init__(self, window: int = 7, model_count: int = 7):
        super().__init__()

        self.window = window  # window for MA and strides
        self.model_count = model_count  # number of days to predict
        self.df_test = None  # other features MA
        self.metrics = {"MAE": [], "RMSE": [],
                        "MAPE": [], "R2": []}  # init metrics dict

    def build_datasets(self, df_train, df_test):

        # creates var; features for test predictions will be taken from here
        self.df_test = pd.DataFrame.copy(df_test)

        # list to store models
        datasets = [float('inf')] * self.model_count
        for i in range(self.model_count):
            # generates feature-sample matrix for X
            X = self.extract_window(df_train, i)

            # extracts
            y = df_train[i + self.window:]['TARGET']

            # appends to ds
            datasets[i] = (X, y)

        assert len(datasets) == self.model_count

        return datasets

    def train_models(self, datasets):

        # trains models; uses catboost
        models = [float('inf')] * self.model_count
        for i in tqdm(range(self.model_count)):
            # rewatch catboost lecture
            model = catboost.CatBoostRegressor(n_estimators=100, verbose=False, allow_writing_files=False,
                                               loss_function='RMSE',
                                               allow_const_label=True)
            model.fit(*datasets[i])
            models[i] = model

        assert len(models) == len(datasets)

        return models

    def predict(self, df_train, models):  # requiers df_train to be passed in
        for i in range(self.model_count):
            # select period to extract features from
            table = self.select_period(df_train, i)
            # flatten features
            row = np.reshape(table, newshape=(
                1, table.size), order='F').squeeze()
            # predicts next value => predict: float
            predict = models[i].predict(row)
            # generates new df (using df_test with precalculated features and new_predict)
            df_predict = self.generate_new_df(self.df_test, predict, i)
            # appends new sample to df
            df_train = pd.concat([df_train, df_predict])
            # process repeats

        # returns predictions
        return df_train[-self.model_count:]['TARGET']


class Model(ModelBuilder):

    def __init__(self):
        super().__init__()
        self.number_of_days_to_predict = 7
        self.number_of_days_for_sliding_window = 5
        self.datasets: List = []
        self.models: List = []

    def pipeline(self, inference=False, df_=pd.DataFrame):
        df_train = df_.copy()
        # load the data

        # split the data
        df_train = self.preprocess_df(df_train)

        if not inference:
            ModelBuilder.__init__(self, window=self.number_of_days_for_sliding_window,
                                  model_count=self.number_of_days_to_predict)
            # creates self.weekly_mean on df_train; MUST BE DONE
            self.extract_weekly_mean(df_train)
            df_train, df_test = self.split_df(df_train)
            df_test = self.enreach_data_1(df_test)

        df_train = self.enreach_data_1(df_train)

        if not inference:
            # create datasets, train models, get predictions
            self.datasets = self.build_datasets(df_train, df_test)
            self.models = self.train_models(self.datasets)

        preds = self.predict(df_train, self.models)

        # display if on inference and save predictions
        if inference:
            self.display_predictions(df_train['TARGET'], preds)

        # display if not on inference and show metrics
        if not inference:
            self.score_model(df_test['TARGET'], preds)
            self.display_predictions(df_test['TARGET'], preds)
            display(pd.DataFrame(self.metrics, index=[0]))

    def main(self, inference=False, df=pd.DataFrame()):
        self.pipeline(inference, df)
