import numpy as np
import pandas as pd
import os
import pprint

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pprint.PrettyPrinter(indent=4)


class ReadData:
    def __init__(self) -> None:
        # 479580b8-30b1-4ace-9da1-77649e3c39ee
        self.base_path = "./data/shops"
        self.columns_to_join_on = ["date", "offer_id", "shop_id"]
        self.info_dict = dict()
        self.d: pd.DataFrame = pd.DataFrame({})
        # data, feature, meta
        self.mode: str = ""

    def _update_dict(self, df: pd.DataFrame):
        isna = df.isna().sum()
        self.info_dict[df['shop_id'][0]] = dict(isna[isna > 0])

    def _join_data_and_feature_df(self, path: str):
        path_shop = os.path.join(self.base_path, path)
        file_names = list(os.walk(path_shop))[0][2]

        for file_name in file_names:
            if self.mode in file_name:
                df = pd.read_csv(os.path.join(path_shop, file_name))
                return df

    def info(self) -> None:
        print(self.df.shape)
        pprint.pprint(self.info_dict)

    def read(self, mode):
        self.mode = mode
        shop_paths = iter(list(os.walk("./data/shops"))[0][1])

        path = next(shop_paths)

        df = self._join_data_and_feature_df(path)
        self._update_dict(df)

        for path in shop_paths:
            df1 = self._join_data_and_feature_df(path)
            self._update_dict(df1)

            df = pd.concat([df, df1], copy=False)

        self.df = df
        return df
