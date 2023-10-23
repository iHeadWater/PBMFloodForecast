import os

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import distance, Point

import definitions
import geopandas as gpd

sl_stas_table: GeoDataFrame = gpd.read_file(
    os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'))
biliu_stas_table = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'), encoding='gbk')


def test_filter_abnormal_sl_and_biliu():
    biliu_his_stas_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    sl_biliu_stas_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    # 碧流河历史数据中，128、138、139、158号站点数据和era5数据出现较大偏差，舍去
    # 松辽委历史数据中，2022年站点数据和era5偏差较大，可能是4、5、9、10月缺测导致


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
                                drop_list.append(range(i - 5, i + 1))
                    table.drop(index=drop_list)
                if 'paravalue' in table.columns:
                    for i in range(0, len(table['paravalue'])):
                        if table['paravalue'][i] > 200:
                            drop_list.append(table['paravalue'][i])
                        if i >= 5:
                            hour_slice = table['paravalue'][i - 5:i + 1].to_numpy()
                            if hour_slice.all() == np.mean(hour_slice):
                                drop_list.append(range(i - 5, i + 1))
                    table.drop(index=drop_list)
                time_df_dict[file.split('_')[0]] = table
    return time_df_dict


def test_filter_data_by_space(time_df_dict):
    neighbor_stas_dict = get_find_neighbor_dict(sl_stas_table, biliu_stas_table)[0]
    gdf_stid_total = get_find_neighbor_dict(sl_stas_table, biliu_stas_table)[1]
    for key in time_df_dict:
        time_drop_list = []
        neighbor_stas = neighbor_stas_dict[key]
        table = time_df_dict[key]
        if len(neighbor_stas) < 12:
            for time in table.index:
                weight_rain = 0
                weight_dis = 0
                for sta in neighbor_stas:
                    weight_rain += table['DRP'][time] / (gdf_stid_total['distance'][gdf_stid_total['STCD'] == sta] ** 2)
                    weight_dis += 1 / (gdf_stid_total['distance'][gdf_stid_total['STCD'] == sta] ** 2)
                interp_rain = weight_rain / weight_dis
                if abs(interp_rain - table['DRP'][time]) > 4:
                    time_drop_list.append(time)
        else:
            for time in table.index:
                rain_time_list = []
                for neighbor in neighbor_stas_dict.keys():
                    rain_time_list.append(time_df_dict[neighbor])
        # Uncompleted


def get_find_neighbor_dict(sl_biliu_gdf, biliu_stbprp_df):
    biliu_stbprp_df = biliu_stbprp_df[biliu_stbprp_df['sttp'] == 2].reset_index().drop(columns=['index'])
    point_list = []
    for i in range(0, len(biliu_stbprp_df)):
        point_x = biliu_stbprp_df['lgtd'][i]
        point_y = biliu_stbprp_df['lttd'][i]
        point = Point(point_x, point_y)
        point_list.append(point)
    gdf_biliu = GeoDataFrame({'STCD': biliu_stbprp_df['stid'], 'STNM': biliu_stbprp_df['stname']}, geometry=point_list)
    sl_biliu_gdf_splited = sl_biliu_gdf[['STCD', 'STNM', 'geometry']]
    # 需要筛选雨量
    gdf_stid_total = GeoDataFrame(pd.concat([gdf_biliu, sl_biliu_gdf_splited], axis=0)).reset_index().drop(columns=['index'])
    neighbor_dict = {}
    for i in range(0, len(gdf_stid_total.geometry)):
        stcd = gdf_stid_total['STCD'][i]
        gdf_stid_total['distance'] = gdf_stid_total.apply(lambda x:
                                                          distance(gdf_stid_total.geometry[i], x.geometry), axis=1)
        nearest_stas = gdf_stid_total[(gdf_stid_total['distance'] > 0) & (gdf_stid_total['distance'] <= 0.2)]
        nearest_stas_list = nearest_stas['STCD'].to_list()
        neighbor_dict[stcd] = nearest_stas_list
    gdf_stid_total = gdf_stid_total.drop(columns=['distance'])
    return neighbor_dict, gdf_stid_total

