import math
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost.sklearn as xgs
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import definitions
import joblib as jl


def test_xgb_find_abnormal_otq():
    era_path = os.path.join(definitions.ROOT_DIR, 'example/era5_xaj/')
    dt_reg = DecisionTreeRegressor()
    biliu_flow_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_rsvr.csv'),
                                engine='c', parse_dates=['TM'])
    date_x = (pd.date_range('2018-1-1 00:00:00', '2022-12-31 23:00:00', freq='H') -
              pd.to_datetime('2000-01-01 00:00:00'))/np.timedelta64(1, 'h')
    rain_y = np.array([])
    for year in range(2018, 2023):
        for month in range(1, 13):
            if month < 10:
                path_era_file = os.path.join(era_path, 'era5_datas_' + str(year) + str(0) + str(month) + '.nc')
            else:
                path_era_file = os.path.join(era_path, 'era5_datas_' + str(year) + str(month) + '.nc')
            era_ds = xr.open_dataset(path_era_file)
            # sro在era5land数据中代表地表径流
            month_rain = era_ds.sel(longitude=122.5, latitude=39.8)['sro']
            rain_y = np.append(rain_y, month_rain)
    X_train, X_test, y_train, y_test = train_test_split(date_x, rain_y, test_size=0.3)
    dt_path = os.path.join(definitions.ROOT_DIR, 'example/dt_reg_test')
    if os.path.exists(dt_path):
        dt_reg = jl.load(dt_path)
    else:
        dt_reg.fit(X=np.expand_dims(X_train, 1), y=np.expand_dims(y_train, 1))
        jl.dump(dt_reg, dt_path)
    pred_era5 = dt_reg.predict(np.expand_dims(X_test, 1))
    r2_era5 = metrics.r2_score(pred_era5, y_test)
    rmse_era5 = math.sqrt(metrics.mean_squared_error(pred_era5, y_test))
    print(r2_era5, rmse_era5)
    predict_range = (biliu_flow_df['TM'][~biliu_flow_df['OTQ'].isna()] - pd.to_datetime('2000-01-01 00:00:00'))/np.timedelta64(1, 'h')
    pred_y = dt_reg.predict(np.expand_dims(predict_range, 1))
    obs_y = biliu_flow_df['OTQ'][~biliu_flow_df['OTQ'].isna()].to_numpy()
    rmse_array = [math.sqrt(metrics.mean_squared_error(pred_y[i:i+10], obs_y[i:i+10])) for i in range(0, len(pred_y)-10)]
    print(rmse_array)



