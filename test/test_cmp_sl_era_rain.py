import datetime
import os

import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from shapely import Point
import geopandas as gpd

import definitions

gdf_biliu_shp: GeoDataFrame = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp'
                                                                               '/碧流河流域.shp'), engine='pyogrio')


# step1: kgo8gd/tnld77/tdi9atr3mir1e3g6
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
    gdf_rain_stations = intersect_rain_stations().reset_index()
    rain_coords = [(point.x, point.y) for point in gdf_rain_stations.geometry]
    era5_dict = get_era5_history_dict(rain_coords, stcd_array=gdf_rain_stations['STCD'])
    sl_df = pd.DataFrame(sl_dict, index=np.arange(2018, 2023, 1)).T
    era5_df = pd.DataFrame(era5_dict, index=np.arange(2018, 2023, 1)).T
    sl_np = sl_df.to_numpy()
    era5_np = era5_df.to_numpy()
    diff_np = np.round((era5_np - sl_np) / sl_np, 3)
    diff_df = pd.DataFrame(data=diff_np, index=sl_df.index, columns=np.arange(2018, 2023, 1))
    sl_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/sl.csv'))
    era5_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_sl.csv'))
    diff_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_sl_diff.csv'))


def test_cmp_biliu_era_rain():
    history_rain_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    biliu_dict = {}
    for root, dirs, files in os.walk(history_rain_path):
        for file in files:
            stcd = file.split('_')[0]
            if '雨量' in file:
                rain_table = pd.read_csv(os.path.join(history_rain_path, file), engine='c', parse_dates=['systemtime'])
                file_yr_list = []
                for year in range(2018, 2023):
                    rain_sum_yr = rain_table['paravalue'][rain_table['systemtime'].dt.year == year].sum()
                    file_yr_list.append(rain_sum_yr)
                biliu_dict[stcd] = file_yr_list
    biliu_stas = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'), encoding='gbk')
    # 在碧流河历史数据st_stbprp_b中，雨量站的sttp是2，水位站是1
    # 玉石水库（152）在st_water_c表中没有数据，故将其从st_stpara_r.CSV、st_stbprp_b.CSV剔除
    # 小宋屯（138）在st_stpara_r同时作为雨量站和水位站存在，但是在st_stbprb_b中只有一个水位站
    # 方便起见将其在st_stbprb_b.CSV中的sttp改成2
    stcd_array = biliu_stas['stid'][biliu_stas['sttp'] == 2].tolist()
    biliu_lons = biliu_stas['lgtd'][biliu_stas['sttp'] == 2].reset_index()['lgtd']
    biliu_lats = biliu_stas['lttd'][biliu_stas['sttp'] == 2].reset_index()['lttd']
    rain_coords = [(biliu_lons[i], biliu_lats[i]) for i in range(0, len(stcd_array))]
    era5_dict = get_era5_history_dict(rain_coords, stcd_array)
    biliu_df = pd.DataFrame(biliu_dict, index=np.arange(2018, 2023, 1)).T
    era5_df = pd.DataFrame(era5_dict, index=np.arange(2018, 2023, 1)).T
    biliu_np = biliu_df.to_numpy()
    era5_np = era5_df.to_numpy()
    diff_np = np.round((era5_np - biliu_np) / biliu_np, 3)
    diff_df = pd.DataFrame(data=diff_np, index=biliu_df.index, columns=np.arange(2018, 2023, 1))
    biliu_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu.csv'))
    era5_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_biliu.csv'))
    diff_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_biliu_diff.csv'))


def get_era5_history_dict(rain_coords, stcd_array):
    era_path = os.path.join(definitions.ROOT_DIR, 'example/era5_xaj/')
    rain_round_coords = [(round(coord[0], 1), round(coord[1], 1)) for coord in rain_coords]
    era5_dict = {}
    for i in range(0, len(rain_round_coords)):
        stcd = stcd_array[i]
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
    return era5_dict


def intersect_rain_stations():
    pp_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/rain_stations.csv'), engine='c').drop(
        columns=['Unnamed: 0'])
    geo_list = []
    stcd_list = []
    stnm_list = []
    for i in range(0, len(pp_df)):
        xc = pp_df['LGTD'][i]
        yc = pp_df['LTTD'][i]
        stcd_list.append(pp_df['STCD'][i])
        stnm_list.append(pp_df['STNM'][i])
        geo_list.append(Point(xc, yc))
    gdf_pps: GeoDataFrame = gpd.GeoDataFrame({'STCD': stcd_list, 'STNM': stnm_list}, geometry=geo_list)
    gdf_rain_stations = gpd.sjoin(gdf_pps, gdf_biliu_shp, 'inner', 'intersects')
    gdf_rain_stations = gdf_rain_stations[~(gdf_rain_stations['STCD'] == '21422950')]
    gdf_rain_stations.to_file(
        os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'))
    return gdf_rain_stations
