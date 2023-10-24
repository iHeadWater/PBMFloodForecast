import os

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import distance, Point

import definitions
import geopandas as gpd

sl_stas_table: GeoDataFrame = gpd.read_file(
    os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'), engine='pyogrio')
biliu_stas_table = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'),
                               encoding='gbk')


def test_filter_abnormal_sl_and_biliu():
    biliu_his_stas_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    sl_biliu_stas_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    # 碧流河历史数据中，128、138、139、158号站点数据和era5数据出现较大偏差，舍去
    # 松辽委历史数据中，2022年站点数据和era5偏差较大，可能是4、5、9、10月缺测导致
    time_df_dict_biliu_his = test_filter_data_by_time(biliu_his_stas_path,
                                                      filter_station_list=[128, 132, 138, 139, 149, 150, 151, 156, 158])
    time_df_dict_sl_biliu = test_filter_data_by_time(sl_biliu_stas_path)
    time_df_dict_sl_biliu.update(time_df_dict_biliu_his)
    space_df_dict = test_filter_data_by_space(time_df_dict_sl_biliu)
    for key in space_df_dict.keys():
        space_df_dict[key].to_csv(
            os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_between_sl_biliu', key + '_filtered.csv'))


def test_filter_data_by_time(data_path, filter_station_list=None):
    if filter_station_list is None:
        filter_station_list = []
    time_df_dict = {}
    test_filtered_by_time_path = os.path.join(definitions.ROOT_DIR, 'example/filtered_data_by_time')
    for dir_name, sub_dir, files in os.walk(data_path):
        for file in files:
            stcd = file.split('_')[0]
            cached_csv_path = os.path.join(test_filtered_by_time_path, stcd + '.csv')
            if (int(stcd) not in filter_station_list) & (~os.path.exists(cached_csv_path)):
                drop_list = []
                csv_path = os.path.join(data_path, file)
                table = pd.read_csv(csv_path, engine='c')
                # 按降雨最大阈值为200和小时雨量一致性过滤索引
                # 松辽委数据不严格按照小时尺度排列，出于简单可以一概按照小时重采样
                if 'DRP' in table.columns:
                    table['TM'] = pd.to_datetime(table['TM'], format='%Y-%m-%d %H:%M:%S')
                    table = table.drop(columns=['Unnamed: 0']).drop(index=table.index[table['DRP'].isna()])
                    # 21422722号站点中出现了2021-4-2 11：36的数据
                    # 整小时数居，再按小时重采样求和，结果不变
                    table = table.set_index('TM').resample('H').sum()
                    cached_time_array = table.index[table['STCD'] != 0].to_numpy()
                    cached_drp_array = table['DRP'][table['STCD'] != 0].to_numpy()
                    table['STCD'] = int(stcd)
                    table['DRP'] = np.nan
                    table['DRP'][cached_time_array] = cached_drp_array
                    table = table.fillna(-1).reset_index()
                    for i in range(0, len(table['DRP'])):
                        if table['DRP'][i] > 200:
                            drop_list.append(i)
                        if i >= 5:
                            hour_slice = table['DRP'][i - 5:i + 1].to_numpy()
                            if hour_slice.all() == np.mean(hour_slice):
                                drop_list.extend(list(range(i - 5, i + 1)))
                    drop_array = np.unique(np.array(drop_list, dtype=int))
                    table = table.drop(index=drop_array)
                    drop_array_minus = table.index[table['DRP'] == -1]
                    table = table.drop(index=drop_array_minus)
                if 'paravalue' in table.columns:
                    for i in range(0, len(table['paravalue'])):
                        if table['paravalue'][i] > 200:
                            drop_list.append(i)
                        if i >= 5:
                            hour_slice = table['paravalue'][i - 5:i + 1].to_numpy()
                            if hour_slice.all() == np.mean(hour_slice):
                                drop_list.extend(list(range(i - 5, i + 1)))
                    drop_array = np.unique(np.array(drop_list, dtype=int))
                    table = table.drop(index=drop_array)
                time_df_dict[stcd] = table
                table.to_csv(cached_csv_path)
            elif (int(stcd) not in filter_station_list) & (os.path.exists(cached_csv_path)):
                table = pd.read_csv(cached_csv_path, engine='c')
                time_df_dict[stcd] = table
    return time_df_dict


def test_filter_data_by_space(time_df_dict):
    neighbor_stas_dict = find_neighbor_dict(sl_stas_table, biliu_stas_table)[0]
    gdf_stid_total = find_neighbor_dict(sl_stas_table, biliu_stas_table)[1]
    space_df_dict = {}
    for key in time_df_dict:
        time_drop_list = []
        neighbor_stas = neighbor_stas_dict[key]
        table = time_df_dict[key]
        if len(neighbor_stas) < 12:
            if 'DRP' in table.columns:
                table = table.set_index('TM')
            if 'paravalue' in table.columns:
                table = table.set_index('systemtime')
            for time in table.index:
                weight_rain = 0
                weight_dis = 0
                for sta in neighbor_stas:
                    point = gdf_stid_total.geometry[gdf_stid_total['STCD'] == sta].values[0]
                    point_self = gdf_stid_total.geometry[gdf_stid_total['STCD'] == key].values[0]
                    dis = distance(point, point_self)
                    if 'DRP' in table.columns:
                        # 疑似因为不同表格时间索引不同导致崩溃
                        weight_rain += table['DRP'][time] / (dis ** 2)
                        weight_dis += 1 / (dis ** 2)
                    if 'paravalue' in table.columns:
                        weight_rain += table['paravalue'][time] / (dis ** 2)
                        weight_dis += 1 / (dis ** 2)
                interp_rain = weight_rain / weight_dis
                if abs(interp_rain - table['DRP'][time]) > 4:
                    time_drop_list.append(time)
            table = table.drop(index=time_drop_list)
        else:
            for time in table.index:
                rain_time_list = []
                for neighbor in neighbor_stas_dict.keys():
                    neighbor_df = time_df_dict[str(neighbor)]
                    if 'DRP' in neighbor_df.columns:
                        rain_time_list.append(neighbor_df['DRP'][time])
                    if 'paravalue' in neighbor_df.columns:
                        rain_time_list.append(neighbor_df['paravalue'][time])
                rain_time_series = pd.Series(rain_time_list)
                quantile_25 = rain_time_series.quantile(q=0.25)
                quantile_75 = rain_time_series.quantile(q=0.75)
                average = rain_time_series.mean()
                MA_Tct = (table['DRP'][time] - average) / (quantile_75 - quantile_25)
                if MA_Tct > 4:
                    time_drop_list.append(time)
            table = table.drop(index=time_drop_list)
        space_df_dict[key] = table
    return space_df_dict


def find_neighbor_dict(sl_biliu_gdf, biliu_stbprp_df):
    # filter_station_list=[128, 132, 138, 139, 149, 150, 151, 156, 158]
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
    gdf_stid_total = GeoDataFrame(pd.concat([gdf_biliu, sl_biliu_gdf_splited], axis=0)).reset_index().drop(
        columns=['index'])
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
