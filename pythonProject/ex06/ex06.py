from unittest.mock import inplace

import pandas as pd
import numpy as np

raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake','Amy'],
            'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
            'age': [42, np.nan, 36, 24, 73],
            'sex' : ['m', np.nan, 'f', 'm', 'f'],
            'preTestScore': [4, np.nan, np.nan, 2, 3],
            'postTestScore': [25, np.nan, np.nan, 62, 70]}
df = pd.DataFrame(raw_data, columns=['first_name', 'last_name',
                                     'age', 'sex', 'preTestScore', 'postTestScore'])
# print(df)
# print(df.isnull().sum()/len(df))
# print(df.dropna())
# print(df)
#
# df_cleaned = df.dropna(how='all')
# print(df_cleaned)
#
# df['location'] = np.nan
# print(df)
# print(df.dropna(axis=1, how='all'))
#
# print('---------------------------')
# print(df.dropna(axis=0, thresh=1))
# print('---------------------------')
# print(df.dropna(thresh=5))
# print('---------------------------')
# print(df.fillna(100))
# print('---------------------------')
# df["preTestScore"].fillna(df["preTestScore"].mean(), inplace=True)
# print(df)
# print('---------------------------')
# print(df.groupby("sex")["postTestScore"].transform("mean"))
# print('---------------------------')
# df["postTestScore"].fillna(df.groupby("sex")["postTestScore"].transform("mean"),
#                            inplace=True)
# print(df)
# print('---------------------------')
