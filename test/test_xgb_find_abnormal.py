import math
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost.sklearn as xgs
from sklearn import metrics
from sklearn.model_selection import train_test_split

import definitions
import joblib as jl


def test_xgb_find_abnormal_otq():
    era_path = os.path.join(definitions.ROOT_DIR, 'example/era5_xaj/')
    xgb_reg = xgs.XGBRegressor()
    biliu_flow_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_rsvr.csv'),
                                engine='c', parse_dates=['TM'])
    date_x = (pd.date_range('2018-1-1 00:00:00', '2020-12-31 23:00:00', freq='H') -
              pd.to_datetime('2000-01-01 00:00:00'))/np.timedelta64(1, 'h')
    rain_y = np.array([])
    for year in range(2018, 2021):
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
    xgb_reg.fit(X=np.expand_dims(X_train, 0), y=np.expand_dims(y_train, 0))
    jl.dump(xgb_reg, os.path.join(definitions.ROOT_DIR, 'example/xgb_reg_test'))
    pred_era5 = xgb_reg.predict(X_test)
    r2_era5 = metrics.r2_score(pred_era5, y_test)
    rmse_era5 = math.sqrt(metrics.mean_squared_error(pred_era5, y_test))
    print(r2_era5, rmse_era5)
    predict_range = (biliu_flow_df['TM'][~biliu_flow_df['OTQ'].isnan()] - pd.to_datetime('2000-01-01 00:00:00'))/np.timedelta64(1, 'h')
    pred_y = xgb_reg.predict(predict_range)
    obs_y = biliu_flow_df['OTQ'][~biliu_flow_df['OTQ'].isnan()]
    r2 = metrics.r2_score(pred_y, obs_y)
    rmse = math.sqrt(metrics.mean_squared_error(pred_y, obs_y))
    print(r2, rmse)



