import os

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import distance

import definitions
import geopandas as gpd


def test_filter_abnormal_sl_and_biliu():
    biliu_his_stas_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    sl_biliu_stas_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    # 碧流河历史数据中，128、138、139、158号站点数据和era5数据出现较大偏差，舍去
    # 松辽委历史数据中，2022年站点数据和era5偏差较大，可能是4、5、9、10月缺测导致
    sl_stas_table: GeoDataFrame = gpd.read_file(
        os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'))
    biliu_stas_table = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'), encoding='gbk')


def test_filter_data_by_time(data_path, filter_station_list):
    time_df_dict = {}
    for dir_name, sub_dir, files in os.walk(data_path):
        for file in files:
            if file.split('_')[0] not in filter_station_list:
                drop_list = []
                csv_path = os.path.join(data_path, file)
                table = pd.read_csv(csv_path, engine='c')
                # 按降雨最大阈值为200和小时雨量一致性过滤索引
                # 松辽委数据不严格按照小时尺度排列，出于简单可以一概按照小时重采样
                if 'DRP' in table.columns:
                    table['TM'] = pd.to_datetime(table['TM'], format='%Y-%m-%d %H:%M%S')
                    table = table.set_index('TM')
                    table = table.resample(freq='H')
                    for i in range(0, len(table['DRP'])):
                        if table['DRP'][i] > 200:
                            drop_list.append(table['DRP'][i])
                        if i >= 5:
                            hour_slice = table['DRP'][i - 5:i + 1].to_numpy()
                            if hour_slice.all() == np.mean(hour_slice):
                                drop_list.append(range(i-5, i+1))
                    table.drop(index=drop_list)
                if 'paravalue' in table.columns:
                    for i in range(0, len(table['paravalue'])):
                        if table['paravalue'][i] > 200:
                            drop_list.append(table['paravalue'][i])
                        if i >= 5:
                            hour_slice = table['paravalue'][i - 5:i + 1].to_numpy()
                            if hour_slice.all() == np.mean(hour_slice):
                                drop_list.append(range(i-5, i+1))
                    table.drop(index=drop_list)
                time_df_dict[file.split('_')[0]] = table
    return time_df_dict


def test_filter_data_by_space(time_df_dict, stas_table):
    for key in time_df_dict:
        time_drop_list = []
        table = time_df_dict[key]
        # 带geometry的GeoDataFrame全是松辽委的，TM为索引
        if 'geometry' in stas_table.columns:
            key_point = stas_table.geometry[stas_table['STCD'] == key]
            stas_table['distance'] = stas_table.apply(lambda x: distance(key_point, x.geometry))
            # 约定相邻的定义为0.2度
            neighbor_stas = stas_table['STCD'][stas_table['distance'] < 0.2]
            neighbor_stas_dict = {}
            for i in range(0, len(neighbor_stas)):
                neighbor_stas_dict[stas_table['STCD'][i]] = stas_table['distance'][i]
            if len(neighbor_stas) < 12:
                for time in table.index:
                    weight_rain = 0
                    weight_dis = 0
                    for sta in neighbor_stas:
                        weight_rain += table['DRP'][time]/(stas_table['distance'][stas_table['STCD'] == sta]**2)
                        weight_dis += 1/(stas_table['distance'][stas_table['STCD'] == sta]**2)
                    interp_rain = weight_rain/weight_dis
                    if abs(interp_rain - table['DRP'][time]) > 4:
                        time_drop_list.append(time)
            else:
                for time in table.index:
                    rain_time_list = []
                    for neighbor in neighbor_stas_dict.keys():
                        rain_time_list.append(time_df_dict[neighbor])
            # Uncompleted


def test_find_neighbor_dicts(station_file_list):
    # Uncompleted
    return 0


def test_shp_to_csv():
    sl_stas_table: GeoDataFrame = gpd.read_file(
        os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'))
    sl_stas_table.to_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_basin_rain_stas.csv'))







