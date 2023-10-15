import datetime
import os

import numpy as np
import pandas as pd
import xarray as xr

import definitions
from test.test_read_rain_stas import intersect_rain_stations


def test_compare_era5_biliu_yr():
    rain_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas/')
    sl_dict = {}
    for root, dirs, files in os.walk(rain_path):
        for file in files:
            stcd = file.split('_')[0]
            rain_table = pd.read_csv(os.path.join(rain_path, file), engine='c', parse_dates=['TM'])
            file_yr_list = []
            for year in range(2018, 2023):
                rain_sum_yr = rain_table['DRP'][rain_table['TM'].dt.year == year].sum()
                file_yr_list.append(rain_sum_yr)
            sl_dict[stcd] = file_yr_list
    era_path = os.path.join(definitions.ROOT_DIR, 'example/era5_xaj/')
    gdf_rain_stations = intersect_rain_stations().reset_index()
    rain_coords = [(point.x, point.y) for point in gdf_rain_stations.geometry]
    rain_round_coords = [(round(coord[0], 1), round(coord[1], 1)) for coord in rain_coords]
    era5_dict = {}
    for i in range(0, len(rain_round_coords)):
        stcd = gdf_rain_stations['STCD'][i]
        coord = rain_round_coords[i]
        year_sum_list = []
        for year in range(2018, 2023):
            year_sum = 0
            for month in range(4, 11):
                if month < 10:
                    path_era_file = os.path.join(era_path, 'era5_datas_' + str(year) + str(0) + str(month) + '.nc')
                else:
                    path_era_file = os.path.join(era_path, 'era5_datas_' + str(year) + str(month) + '.nc')
                era_ds = xr.open_dataset(path_era_file)
                # tp在era5数据中代表总降雨
                month_rain = era_ds.sel(longitude=coord[0], latitude=coord[1])['tp']
                # 在这里有日期误差（每天0点数据是昨天一天的累积），但涉及到一年尺度，误差不大，可以容忍
                month_rain_daily = month_rain.loc[month_rain.time.dt.time == datetime.time(0, 0)]
                # era5数据单位是m，所以要*1000
                month_rain_sum = (month_rain_daily.sum().to_numpy()) * 1000
                year_sum += month_rain_sum
            year_sum_list.append(year_sum)
        era5_dict[stcd] = year_sum_list
    sl_df = pd.DataFrame(sl_dict, index=np.arange(2018, 2023, 1)).T
    era5_df = pd.DataFrame(era5_dict, index=np.arange(2018, 2023, 1)).T
    sl_np = sl_df.to_numpy()
    era5_np = era5_df.to_numpy()
    diff_np = np.round((era5_np - sl_np)/sl_np, 3)
    diff_df = pd.DataFrame(data=diff_np, index=sl_df.index, columns=np.arange(2018, 2023, 1))
    sl_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/sl.csv'))
    era5_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/era5.csv'))
    diff_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_sl_diff.csv'))
