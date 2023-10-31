import os
import shutil

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import distance, Point

import definitions
import geopandas as gpd

from test.test_read_rain_stas import voronoi_from_shp

sl_stas_table: GeoDataFrame = gpd.read_file(
    os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'), engine='pyogrio')
biliu_stas_table = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'),
                               encoding='gbk')
gdf_biliu_shp: GeoDataFrame = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp'
                                                                               '/碧流河流域.shp'), engine='pyogrio')


def test_filter_abnormal_sl_and_biliu():
    biliu_his_stas_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    sl_biliu_stas_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    # 碧流河历史数据中，128、138、139、158号站点数据和era5数据出现较大偏差，舍去
    # 松辽委历史数据中，2022年站点数据和era5偏差较大，可能是4、5、9、10月缺测导致
    filter_station_list = [128, 138, 139, 158]
    time_df_dict_biliu_his = test_filter_data_by_time(biliu_his_stas_path, filter_station_list)
    time_df_dict_sl_biliu = test_filter_data_by_time(sl_biliu_stas_path)
    time_df_dict_sl_biliu.update(time_df_dict_biliu_his)
    space_df_dict = test_filter_data_by_space(time_df_dict_sl_biliu, filter_station_list)
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
            feature = file.split('_')[1]
            cached_csv_path = os.path.join(test_filtered_by_time_path, stcd + '.csv')
            if (int(stcd) not in filter_station_list) & (~os.path.exists(cached_csv_path)) & (feature != '水位'):
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
            elif (int(stcd) not in filter_station_list) & (os.path.exists(cached_csv_path)) & (feature != '水位'):
                table = pd.read_csv(cached_csv_path, engine='c')
                time_df_dict[stcd] = table
    return time_df_dict


def test_filter_data_by_space(time_df_dict, filter_station_list):
    neighbor_stas_dict = find_neighbor_dict(sl_stas_table, biliu_stas_table, filter_station_list)[0]
    gdf_stid_total = find_neighbor_dict(sl_stas_table, biliu_stas_table, filter_station_list)[1]
    space_df_dict = {}
    for key in time_df_dict:
        time_drop_list = []
        neighbor_stas = neighbor_stas_dict[key]
        table = time_df_dict[key]
        if 'DRP' in table.columns:
            table = table.set_index('TM')
        if 'paravalue' in table.columns:
            table = table.set_index('systemtime')
        for time in table.index:
            rain_time_dict = {}
            for neighbor in neighbor_stas:
                neighbor_df = time_df_dict[str(neighbor)]
                if 'DRP' in neighbor_df.columns:
                    neighbor_df = neighbor_df.set_index('TM')
                    if time in neighbor_df.index:
                        rain_time_dict[str(neighbor)] = neighbor_df['DRP'][time]
                if 'paravalue' in neighbor_df.columns:
                    neighbor_df = neighbor_df.set_index('systemtime')
                    if time in neighbor_df.index:
                        rain_time_dict[str(neighbor)] = neighbor_df['paravalue'][time]
            if len(rain_time_dict) == 0:
                continue
            elif 0 < len(rain_time_dict) < 12:
                weight_rain = 0
                weight_dis = 0
                for sta in rain_time_dict.keys():
                    point = gdf_stid_total.geometry[gdf_stid_total['STCD'] == str(sta)].values[0]
                    point_self = gdf_stid_total.geometry[gdf_stid_total['STCD'] == str(key)].values[0]
                    dis = distance(point, point_self)
                    if 'DRP' in table.columns:
                        weight_rain += table['DRP'][time] / (dis ** 2)
                        weight_dis += 1 / (dis ** 2)
                    elif 'paravalue' in table.columns:
                        weight_rain += table['paravalue'][time] / (dis ** 2)
                        weight_dis += 1 / (dis ** 2)
                interp_rain = weight_rain / weight_dis
                if 'DRP' in table.columns:
                    if abs(interp_rain - table['DRP'][time]) > 4:
                        time_drop_list.append(time)
                elif 'paravalue' in table.columns:
                    if abs(interp_rain - table['paravalue'][time]) > 4:
                        time_drop_list.append(time)
            elif len(rain_time_dict) >= 12:
                rain_time_series = pd.Series(rain_time_dict.values())
                quantile_25 = rain_time_series.quantile(q=0.25)
                quantile_75 = rain_time_series.quantile(q=0.75)
                average = rain_time_series.mean()
                if 'DRP' in table.columns:
                    MA_Tct = (table['DRP'][time] - average) / (quantile_75 - quantile_25)
                    if MA_Tct > 4:
                        time_drop_list.append(time)
                elif 'paravalue' in table.columns:
                    MA_Tct = (table['paravalue'][time] - average) / (quantile_75 - quantile_25)
                    if MA_Tct > 4:
                        time_drop_list.append(time)
        table = table.drop(index=time_drop_list).drop(columns=['Unnamed: 0'])
        space_df_dict[key] = table
    return space_df_dict


def find_neighbor_dict(sl_biliu_gdf, biliu_stbprp_df, filter_station_list):
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
    gdf_stid_total = GeoDataFrame(pd.concat([gdf_biliu, sl_biliu_gdf_splited], axis=0))
    gdf_stid_total = gdf_stid_total.set_index('STCD').drop(index=filter_station_list).reset_index()
    gdf_stid_total['STCD'] = gdf_stid_total['STCD'].astype('str')
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


def get_voronoi_total():
    node_shp = os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas_total.shp')
    dup_basin_shp = os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/碧流河流域_副本.shp')
    origin_basin_shp = os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/碧流河流域.shp')
    if not os.path.exists(node_shp):
        shutil.copyfile(origin_basin_shp, dup_basin_shp)
        gdf_stid_total = find_neighbor_dict(sl_stas_table, biliu_stas_table, filter_station_list=[128, 138, 139, 158])[1]
        gdf_stid_total.to_file(node_shp)
    voronoi_from_shp(src=node_shp, des=dup_basin_shp)
    voronoi_gdf = gpd.read_file(dup_basin_shp, engine='pyogrio')
    return voronoi_gdf


def test_rain_average_filtered(start_date='2014-01-01 00:00:00', end_date='2022-09-01 00:00:00'):
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')
    voronoi_gdf = get_voronoi_total()
    voronoi_gdf['real_area'] = voronoi_gdf.apply(lambda x: x.geometry.area*12100, axis=1)
    rain_path = os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_between_sl_biliu')
    table_dict = {}
    for root, dirs, files in os.walk(rain_path):
        for file in files:
            stcd = file.split('_')[0]
            rain_table = pd.read_csv(os.path.join(rain_path, file), engine='c')
            if 'TM' in rain_table.columns:
                rain_table['TM'] = pd.to_datetime(rain_table['TM'])
            elif 'systemtime' in rain_table.columns:
                rain_table['systemtime'] = pd.to_datetime(rain_table['systemtime'])
            table_dict[stcd] = rain_table
    # 参差不齐，不能直接按照长时间序列选择，只能一个个时间索引去找，看哪个站有数据，再做平均
    rain_aver_dict = {}
    for time in pd.date_range(start_date, end_date, freq='H'):
        time_rain_records = {}
        for stcd in table_dict.keys():
            rain_table = table_dict[stcd]
            if 'DRP' in rain_table.columns:
                if time in rain_table['TM']:
                    drp = rain_table['DRP'][rain_table['TM'] == time]
                    time_rain_records[stcd] = drp
                else:
                    drp = 0
                    time_rain_records[stcd] = drp
            elif 'paravalue' in rain_table.columns:
                if time in rain_table['systemtime']:
                    drp = rain_table['paravalue'][rain_table['systemtime'] == time]
                    time_rain_records[stcd] = drp
                else:
                    drp = 0
                    time_rain_records[stcd] = drp
            else:
                continue
        rain_aver = 0
        for stcd in time_rain_records.keys():
            voronoi_gdf['STCD'] = voronoi_gdf['STCD'].astype('str')
            rain_aver += time_rain_records[stcd] * voronoi_gdf['real_area'][voronoi_gdf['STCD'] == stcd].values[0] / gdf_biliu_shp['area'][0]
        rain_aver_dict[time] = rain_aver
    rain_aver_df = pd.DataFrame({'TM': rain_aver_dict.keys(), 'rain': rain_aver_dict.values()})
    rain_aver_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_average.csv'))
    return rain_aver_dict
