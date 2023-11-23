import os
import shutil

import geopandas as gpd
import hydromodel.models.xaj
import numpy as np
import pandas as pd
import whitebox
from geopandas import GeoDataFrame
from hydromodel.calibrate.calibrate_ga import calibrate_by_ga
from hydromodel.utils.dmca_esr import step1_step2_tr_and_fluctuations_timeseries, step3_core_identification, \
    step4_end_rain_events, \
    step5_beginning_rain_events, step6_checks_on_rain_events, step7_end_flow_events, step8_beginning_flow_events, \
    step9_checks_on_flow_events, step10_checks_on_overlapping_events
from hydromodel.utils.stat import statRmse
from matplotlib import pyplot as plt
from pandas import DataFrame
from shapely import distance, Point

import definitions

sl_stas_table: GeoDataFrame = gpd.read_file(
    os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'), engine='pyogrio')
biliu_stas_table = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'),
                               encoding='gbk')
gdf_biliu_shp: GeoDataFrame = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/碧流河流域.shp'),
                                            engine='pyogrio')
# 碧流河历史数据中，128、138、139、158号站点数据和era5数据出现较大偏差，舍去
# 松辽委历史数据中，2022年站点数据和era5偏差较大，可能是4、5、9、10月缺测导致
# 碧流河历史数据中，126、127、129、130、133、141、142、154出现过万极值，需要另行考虑或直接剔除
# 134、137、143、144出现千级极值，需要再筛选
filter_station_list = [128, 138, 139, 158]


class my:
    data_dir = '.'

    @classmethod
    def my_callback(cls, value):
        if not "*" in value and not "%" in value:
            print(value)
        if "Elapsed Time" in value:
            print('--------------')

    @classmethod
    def my_callback_home(cls, value):
        if not "*" in value and not "%" in value:
            print(value)
        if "Output file written" in value:
            os.chdir(cls.data_dir)


def voronoi_from_shp(src, des, data_dir='.'):
    my.data_dir = os.path.abspath(data_dir)
    src = os.path.abspath(src)
    des = os.path.abspath(des)
    wbt = whitebox.WhiteboxTools()
    wbt.voronoi_diagram(src, des, callback=my.my_callback)


def test_calc_filter_station_list():
    # 可以和之前比较的方法接起来而不是读csv
    era5_sl_diff_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_sl_diff.csv')).rename(
        columns={'Unnamed: 0': 'STCD'})
    era5_biliu_diff_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_biliu_diff.csv')).rename(
        columns={'Unnamed: 0': 'STCD'})
    biliu_hourly_splited_path = os.path.join(definitions.ROOT_DIR,
                                             'example/biliu_history_data/history_data_splited_hourly')
    biliu_hourly_filtered_path = os.path.join(definitions.ROOT_DIR, 'example/filtered_data_by_time')
    sl_hourly_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    station_vari_dict = {}
    station_vari_dict_by_time = {}
    filter_station_list = []
    for i in range(0, len(era5_biliu_diff_df)):
        if np.inf in era5_biliu_diff_df.iloc[i].to_numpy():
            filter_station_list.append(era5_biliu_diff_df['STCD'][i])
    for i in range(0, len(era5_sl_diff_df)):
        if np.inf in era5_sl_diff_df.iloc[i].to_numpy():
            filter_station_list.append(era5_sl_diff_df['STCD'][i])
    for dir_name, sub_dirs, files in os.walk(biliu_hourly_splited_path):
        for file in files:
            stcd = file.split('_')[0]
            csv_path = os.path.join(biliu_hourly_splited_path, file)
            biliu_df = pd.read_csv(csv_path, engine='c')
            if int(stcd) not in filter_station_list:
                para_std = biliu_df['paravalue'].std()
                para_aver = biliu_df['paravalue'].mean()
                vari_corr = para_std / para_aver
                station_vari_dict[stcd] = vari_corr
                if vari_corr > 3:
                    filter_station_list.append(int(stcd))
    for dir_name, sub_dirs, files in os.walk(sl_hourly_path):
        for file in files:
            stcd = file.split('_')[0]
            csv_path = os.path.join(sl_hourly_path, file)
            sl_df = pd.read_csv(csv_path, engine='c')
            if int(stcd) not in filter_station_list:
                para_std = sl_df['DRP'].std()
                para_aver = sl_df['DRP'].mean()
                vari_corr = para_std / para_aver
                station_vari_dict[stcd] = vari_corr
                if vari_corr > 3:
                    filter_station_list.append(int(stcd))
    for dir_name, sub_dirs, files in os.walk(biliu_hourly_filtered_path):
        for file in files:
            stcd = file.split('.')[0]
            csv_path = os.path.join(biliu_hourly_filtered_path, file)
            data_df = pd.read_csv(csv_path, engine='c')
            if int(stcd) not in filter_station_list:
                if 'DRP' in data_df.columns:
                    para_std = data_df['DRP'].std()
                    para_aver = data_df['DRP'].mean()
                    vari_corr = para_std / para_aver
                    station_vari_dict_by_time[stcd] = vari_corr
                    if vari_corr > 3:
                        filter_station_list.append(int(stcd))
                elif 'paravalue' in data_df.columns:
                    para_std = data_df['paravalue'].std()
                    para_aver = data_df['paravalue'].mean()
                    vari_corr = para_std / para_aver
                    station_vari_dict_by_time[stcd] = vari_corr
                    if vari_corr > 3:
                        filter_station_list.append(int(stcd))
    print(filter_station_list)
    print(station_vari_dict)
    print(station_vari_dict_by_time)
    return filter_station_list


def test_filter_abnormal_sl_and_biliu():
    biliu_his_stas_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    sl_biliu_stas_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    time_df_dict_biliu_his = get_filter_data_by_time(biliu_his_stas_path, filter_station_list)
    time_df_dict_sl_biliu = get_filter_data_by_time(sl_biliu_stas_path)
    time_df_dict_sl_biliu.update(time_df_dict_biliu_his)
    space_df_dict = get_filter_data_by_space(time_df_dict_sl_biliu, filter_station_list)
    for key in space_df_dict.keys():
        space_df_dict[key].to_csv(
            os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_between_sl_biliu', key + '_filtered.csv'))


def get_filter_data_by_time(data_path, filter_list=None):
    if filter_list is None:
        filter_list = []
    time_df_dict = {}
    test_filtered_by_time_path = os.path.join(definitions.ROOT_DIR, 'example/filtered_data_by_time')
    for dir_name, sub_dir, files in os.walk(data_path):
        for file in files:
            stcd = file.split('_')[0]
            feature = file.split('_')[1]
            cached_csv_path = os.path.join(test_filtered_by_time_path, stcd + '.csv')
            if (int(stcd) not in filter_list) & (~os.path.exists(cached_csv_path)) & (feature != '水位'):
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
            elif (int(stcd) not in filter_list) & (os.path.exists(cached_csv_path)) & (feature != '水位'):
                table = pd.read_csv(cached_csv_path, engine='c')
                time_df_dict[stcd] = table
    return time_df_dict


def get_filter_data_by_space(time_df_dict, filter_list):
    neighbor_stas_dict = find_neighbor_dict(sl_stas_table, biliu_stas_table, filter_list)[0]
    gdf_stid_total = find_neighbor_dict(sl_stas_table, biliu_stas_table, filter_list)[1]
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


def find_neighbor_dict(sl_biliu_gdf, biliu_stbprp_df, filter_list):
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
    gdf_stid_total = gdf_stid_total.set_index('STCD').drop(index=filter_list).reset_index()
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
        gdf_stid_total = find_neighbor_dict(sl_stas_table, biliu_stas_table, filter_list=filter_station_list)[1]
        gdf_stid_total.to_file(node_shp)
    voronoi_from_shp(src=node_shp, des=dup_basin_shp)
    voronoi_gdf = gpd.read_file(dup_basin_shp, engine='pyogrio')
    return voronoi_gdf


def test_rain_average_filtered(start_date='2014-01-01 00:00:00', end_date='2022-09-01 00:00:00'):
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')
    voronoi_gdf = get_voronoi_total()
    voronoi_gdf['real_area'] = voronoi_gdf.apply(lambda x: x.geometry.area * 12100, axis=1)
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
                if time in rain_table['TM'].to_numpy():
                    drp = rain_table['DRP'][rain_table['TM'] == time]
                    time_rain_records[stcd] = drp.values[0]
                else:
                    drp = 0
                    time_rain_records[stcd] = drp
            elif 'paravalue' in rain_table.columns:
                if time in rain_table['systemtime'].to_numpy():
                    drp = rain_table['paravalue'][rain_table['systemtime'] == time]
                    time_rain_records[stcd] = drp.values[0]
                else:
                    drp = 0
                    time_rain_records[stcd] = drp
            else:
                continue
        rain_aver = 0
        for stcd in time_rain_records.keys():
            voronoi_gdf['STCD'] = voronoi_gdf['STCD'].astype('str')
            rain_aver += time_rain_records[stcd] * voronoi_gdf['real_area'][voronoi_gdf['STCD'] == stcd].values[0] / \
                         gdf_biliu_shp['area'][0]
        rain_aver_dict[time] = rain_aver
    rain_aver_df = pd.DataFrame({'TM': rain_aver_dict.keys(), 'rain': rain_aver_dict.values()})
    rain_aver_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_average.csv'))
    return rain_aver_dict


def get_infer_inq():
    inq_csv_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_inq_interpolated.csv')
    if os.path.exists(inq_csv_path):
        new_df = pd.read_csv(inq_csv_path, engine='c', parse_dates=['TM']).set_index('TM')
    else:
        biliu_flow_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_rsvr.csv'),
                                    engine='c', parse_dates=['TM'])
        biliu_area = gdf_biliu_shp.geometry[0].area * 12100
        biliu_flow_df: DataFrame = biliu_flow_df.fillna(-1)
        inq_array = biliu_flow_df['INQ'].to_numpy()
        otq_array = biliu_flow_df['OTQ'].to_numpy()
        w_array = biliu_flow_df['W'].to_numpy()
        tm_array = biliu_flow_df['TM'].to_numpy()
        for i in range(1, len(biliu_flow_df)):
            if (int(inq_array[i]) == -1) & (int(otq_array[i]) != -1):
                # TypeError: unsupported operand type(s) for -: 'str' and 'str'
                time_div = np.timedelta64(tm_array[i] - tm_array[i - 1]) / np.timedelta64(1, 'h')
                inq_array[i] = otq_array[i] + (w_array[i] - w_array[i - 1]) / time_div
        # 还要根据时间间隔插值
        new_df = pd.DataFrame({'TM': tm_array, 'INQ': inq_array, 'OTQ': otq_array})
        new_df = new_df.set_index('TM').resample('H').interpolate()
        # 流量单位转换
        new_df['INQ_mm/h'] = new_df['INQ'].apply(lambda x: x * 3.6 / biliu_area)
        new_df.to_csv(inq_csv_path)
    return new_df['INQ'], new_df['INQ_mm/h']


def biliu_rain_flow_division():
    # rain和flow之间的索引要尽量“对齐”
    # 2014.1.1 00:00:00-2022.9.1 00:00:00
    filtered_rain_aver_df = (pd.read_csv(os.path.join(definitions.ROOT_DIR,
                                                      'example/filtered_rain_average.csv'), engine='c').
                             set_index('TM').drop(columns=['Unnamed: 0']))
    filtered_rain_aver_array = filtered_rain_aver_df['rain'].to_numpy()
    flow_m3_s = (get_infer_inq()[0])[filtered_rain_aver_df.index]
    flow_mm_h = (get_infer_inq()[1])[filtered_rain_aver_df.index]
    time = filtered_rain_aver_df.index
    rain_min = 0.01
    max_window = 100
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = step1_step2_tr_and_fluctuations_timeseries(
        filtered_rain_aver_array, flow_mm_h,
        rain_min,
        max_window)
    beginning_core, end_core = step3_core_identification(fluct_bivariate_Tr)
    end_rain = step4_end_rain_events(beginning_core, end_core, filtered_rain_aver_array, fluct_rain_Tr, rain_min)
    beginning_rain = step5_beginning_rain_events(beginning_core, end_rain, filtered_rain_aver_array, fluct_rain_Tr,
                                                 rain_min)
    beginning_rain_checked, end_rain_checked, beginning_core, end_core = step6_checks_on_rain_events(beginning_rain,
                                                                                                     end_rain,
                                                                                                     filtered_rain_aver_array,
                                                                                                     rain_min,
                                                                                                     beginning_core,
                                                                                                     end_core)
    end_flow = step7_end_flow_events(end_rain_checked, beginning_core, end_core, filtered_rain_aver_array,
                                     fluct_rain_Tr, fluct_flow_Tr, Tr)
    beginning_flow = step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, filtered_rain_aver_array,
                                                 beginning_core,
                                                 fluct_rain_Tr, fluct_flow_Tr)
    beginning_flow_checked, end_flow_checked = step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked,
                                                                           beginning_flow,
                                                                           end_flow, fluct_flow_Tr)
    BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW = step10_checks_on_overlapping_events(beginning_rain_checked,
                                                                                             end_rain_checked,
                                                                                             beginning_flow_checked,
                                                                                             end_flow_checked, time)
    print(len(BEGINNING_RAIN), len(END_RAIN), len(BEGINNING_FLOW), len(END_FLOW))
    print('_________________________')
    print(BEGINNING_RAIN, END_RAIN)
    print('_________________________')
    print(BEGINNING_FLOW, END_FLOW)
    # 从自动划分结果里手动选几个场次
    session_times = [('2017/8/1 15:00:00', '2017/8/7 07:00:00'), ('2018/8/19 12:00:00', '2018/8/23 09:00:00'),
                     ('2020/8/31 04:00:00', '2020/9/4 15:00:00'), ('2022/7/6 10:00:00', '2022/7/10 00:00:00'),
                     ('2022/8/6 10:00:00', '2022/8/11 00:00:00')]
    session_df_list = []
    for session in session_times:
        start_time = pd.to_datetime(session[0])
        end_time = pd.to_datetime(session[1])
        filtered_rain_aver_df.index = pd.to_datetime(filtered_rain_aver_df.index)
        rain_session = filtered_rain_aver_df[start_time: end_time]
        flow_session_mm_h = flow_mm_h[start_time: end_time]
        flow_session_m3_s = flow_m3_s[start_time: end_time]
        session_df = pd.DataFrame(
            {'TM': pd.date_range(start_time, end_time, freq='H'), 'RAIN_SESSION': rain_session.to_numpy().flatten()
                , 'FLOW_SESSION_MM_H': flow_session_mm_h.to_numpy(), 'FLOW_SESSION_M3_S': flow_session_m3_s.to_numpy()})
        session_df_list.append(session_df)
    return session_df_list


def get_deap_dir_by_session(df: DataFrame):
    top_deap_dir = os.path.join(definitions.ROOT_DIR, 'example/deap_dir/')
    time_index = df.index[0].strftime('%Y-%m-%d-%H-%M-%S')
    deap_dir = os.path.join(top_deap_dir, time_index)
    if not os.path.exists(deap_dir):
        os.makedirs(deap_dir)
    return deap_dir


# need fusion with the last test
def test_calibrate_flow():
    # pet_df含有潜在蒸散发
    pet_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_data/pet_calc/PET_result.CSV'), engine='c',
                         parse_dates=['time']).set_index('time')
    session_df_list = biliu_rain_flow_division()
    for session_df in session_df_list:
        # session_df 含有雨和洪
        session_df = session_df.set_index('TM')
        deap_dir = get_deap_dir_by_session(session_df)
        session_pet = pet_df.loc[session_df.index].to_numpy().flatten()
        calibrate_df = pd.DataFrame({'PRCP': session_df['RAIN_SESSION'].to_numpy(), 'PET': session_pet,
                                     'streamflow': session_df['FLOW_SESSION_MM_H'].to_numpy()})
        calibrate_np = calibrate_df.to_numpy()
        calibrate_np = np.expand_dims(calibrate_np, axis=0)
        calibrate_np = np.swapaxes(calibrate_np, 0, 1)
        observed_output = np.expand_dims(calibrate_np[:, :, -1], axis=0)
        observed_output = np.swapaxes(observed_output, 0, 1)
        pop = calibrate_by_ga(input_data=calibrate_np[:, :, 0:2], observed_output=observed_output, deap_dir=deap_dir,
                              warmup_length=24)
        print(pop)


def test_compare_paras():
    # 遗传算法是按照mm/h率定的
    '''
    test_session_times = [('2017/8/1 15:00:00', '2017/8/7 07:00:00'), ('2018/8/19 12:00:00', '2018/8/23 09:00:00'),
                         ('2020/8/31 04:00:00', '2020/9/4 15:00:00'), ('2022/7/6 10:00:00', '2022/7/10 00:00:00'),
                         ('2022/8/6 10:00:00', '2022/8/11 00:00:00')]
    filtered_rain_aver_df = (pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_average.csv'),
                                         engine='c', parse_dates=['TM']).set_index('TM').drop(columns=['Unnamed: 0']))
    flow_m3_s = (get_infer_inq()[0])[filtered_rain_aver_df.index]
    pet_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_data/pet_calc/PET_result.CSV'), engine='c',
                         parse_dates=['time']).set_index('time')
    test_session_dict = {}
    for test_session_time in test_session_times:
        start_time = pd.to_datetime(test_session_time[0])
        end_time = pd.to_datetime(test_session_time[1])
        rain_session = filtered_rain_aver_df[start_time: end_time]
        session_pet = pet_df.loc[start_time: end_time].to_numpy().flatten()
        flow_session_m3_s = flow_m3_s[start_time: end_time]
        test_session_df = pd.DataFrame({'RAIN_SESSION': rain_session.to_numpy().flatten(),
                                        'PET': session_pet,
                                        'FLOW_SESSION_M3_S': flow_session_m3_s.to_numpy()})
        test_session_np = test_session_df.to_numpy()
        test_session_np = np.expand_dims(test_session_np, axis=0)
        test_session_np = np.swapaxes(test_session_np, 0, 1)
        test_session_dict[start_time.strftime('%Y-%m-%d-%H-%M-%S')] = test_session_np
    '''
    pet_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_data/pet_calc/PET_result.CSV'), engine='c',
                         parse_dates=['time']).set_index('time')
    sessions_list = biliu_rain_flow_division()
    for session_df in sessions_list:
        session_df = session_df.set_index('TM')
        deap_dir = get_deap_dir_by_session(session_df)
        time_dir = deap_dir.split('/')[-1]
        pkl_epoch5_path = os.path.join(deap_dir, 'epoch5.pkl')
        pkl_xaj = np.load(pkl_epoch5_path, allow_pickle=True)
        warmup_length = 24
        pet_session = pet_df.loc[session_df.index]
        session_df = pd.concat([session_df['RAIN_SESSION'], pet_session, session_df['FLOW_SESSION_MM_H'], session_df['FLOW_SESSION_M3_S']], axis=1)
        session_np = session_df.to_numpy()
        session_np = np.expand_dims(session_np, axis=1)
        qsim, es = hydromodel.models.xaj.xaj(p_and_e=session_np[:, :, 0:2],
                                             params=np.array(pkl_xaj['halloffame'][0]).reshape(1, -1),
                                             warmup_length=warmup_length, name='xaj_mz')
        qsim = qsim * 2097000 / 3600
        y_flow_obs = session_np[:, :, -1].flatten()[warmup_length:]
        rmse = statRmse(qsim.flatten(), y_flow_obs)
        x = session_df.index[warmup_length:]
        rain_session = session_df['RAIN_SESSION']
        y_rain_obs = rain_session.to_numpy().flatten()
        fig, ax = plt.subplots(figsize=(16, 12))
        p = ax.twinx()
        ax.bar(x, y_rain_obs[warmup_length:], color='red', alpha=0.6, width=0.04)
        ax.set_ylabel('rain(mm)')
        ax.invert_yaxis()
        p.plot(x, y_flow_obs, color='green', linewidth=2)
        p.plot(x, qsim.flatten(), color='yellow', linewidth=2)
        p.set_ylabel('flow(m3/s)')
        plt.savefig(os.path.join(definitions.ROOT_DIR, 'example/deap_dir/calibrated_xaj_cmp', time_dir+'.png'))
        print(rmse)
