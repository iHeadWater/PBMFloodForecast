import os
import shutil

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from hydromodel.utils.dmca_esr import step1_step2_tr_and_fluctuations_timeseries, step3_core_identification, step4_end_rain_events, \
    step5_beginning_rain_events, step6_checks_on_rain_events, step7_end_flow_events, step8_beginning_flow_events, \
    step9_checks_on_flow_events, step10_checks_on_overlapping_events
from pandas import DataFrame
from shapely import distance, Point
from xaj.calibrate_ga_xaj_bmi import calibrate_by_ga

import definitions
import geopandas as gpd
import whitebox
import matplotlib.pyplot as plt


sl_stas_table: GeoDataFrame = gpd.read_file(
    os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/biliu_basin_rain_stas.shp'), engine='pyogrio')
biliu_stas_table = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/st_stbprp_b.CSV'),
                               encoding='gbk')
gdf_biliu_shp: GeoDataFrame = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/碧流河流域.shp'), engine='pyogrio')
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
    era5_sl_diff_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_sl_diff.csv')).rename(columns={'Unnamed: 0': 'STCD'})
    era5_biliu_diff_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_biliu_diff.csv')).rename(columns={'Unnamed: 0': 'STCD'})
    biliu_hourly_splited_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
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
                vari_corr = para_std/para_aver
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
                vari_corr = para_std/para_aver
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
                    vari_corr = para_std/para_aver
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
            rain_aver += time_rain_records[stcd] * voronoi_gdf['real_area'][voronoi_gdf['STCD'] == stcd].values[0] / gdf_biliu_shp['area'][0]
        rain_aver_dict[time] = rain_aver
    rain_aver_df = pd.DataFrame({'TM': rain_aver_dict.keys(), 'rain': rain_aver_dict.values()})
    rain_aver_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_average.csv'))
    return rain_aver_dict


def get_infer_inq():
    inq_csv_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_inq_interpolated.csv')
    if os.path.exists(inq_csv_path):
        new_df = pd.read_csv(inq_csv_path, engine='c').set_index('TM')
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


def test_biliu_rain_flow_division():
    # rain和flow之间的索引要尽量“对齐”
    # 2014.1.1 00:00:00-2022.9.1 00:00:00
    filtered_rain_aver_df = (pd.read_csv(os.path.join(definitions.ROOT_DIR,
                                                      'example/filtered_rain_average.csv'), engine='c').
                             set_index('TM').drop(columns=['Unnamed: 0']))
    filtered_rain_aver_array = filtered_rain_aver_df['rain'].to_numpy()
    # flow_m3_s = (get_infer_inq()[0])[filtered_rain_aver_df.index]
    flow_mm_h = (get_infer_inq()[1])[filtered_rain_aver_df.index]
    time = filtered_rain_aver_df.index
    rain_min = 0.01
    max_window = 100
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = step1_step2_tr_and_fluctuations_timeseries(filtered_rain_aver_array, flow_mm_h,
                                                                                                      rain_min,
                                                                                                      max_window)
    beginning_core, end_core = step3_core_identification(fluct_bivariate_Tr)
    end_rain = step4_end_rain_events(beginning_core, end_core, filtered_rain_aver_array, fluct_rain_Tr, rain_min)
    beginning_rain = step5_beginning_rain_events(beginning_core, end_rain, filtered_rain_aver_array, fluct_rain_Tr, rain_min)
    beginning_rain_checked, end_rain_checked, beginning_core, end_core = step6_checks_on_rain_events(beginning_rain,
                                                                                                     end_rain, filtered_rain_aver_array,
                                                                                                     rain_min,
                                                                                                     beginning_core,
                                                                                                     end_core)
    end_flow = step7_end_flow_events(end_rain_checked, beginning_core, end_core, filtered_rain_aver_array, fluct_rain_Tr, fluct_flow_Tr, Tr)
    beginning_flow = step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, filtered_rain_aver_array, beginning_core,
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
    wjy_calibrate_time = pd.read_excel(os.path.join(definitions.ROOT_DIR, 'example/洪水率定时间.xlsx'))
    wjy_calibrate_time['starttime'] = pd.to_datetime(wjy_calibrate_time['starttime'], format='%Y/%m/%d %H:%M:%S')
    wjy_calibrate_time['endtime'] = pd.to_datetime(wjy_calibrate_time['endtime'], format='%Y/%m/%d %H:%M:%S')
    for i in range(14, 25):
        start_time = wjy_calibrate_time['starttime'][i]
        end_time = wjy_calibrate_time['endtime'][i]
        x = pd.date_range(start_time, end_time, freq='H')
        fig, ax = plt.subplots(figsize=(9, 6))
        p = ax.twinx()
        filtered_rain_aver_df.index = pd.to_datetime(filtered_rain_aver_df.index)
        flow_mm_h.index = pd.to_datetime(flow_mm_h.index)
        y_rain = filtered_rain_aver_df[start_time: end_time]
        y_flow = flow_mm_h[start_time:end_time]
        ax.bar(x, y_rain.to_numpy().flatten(), color='red', edgecolor='k', alpha=0.6, width=0.04)
        ax.set_ylabel('rain(mm)')
        ax.invert_yaxis()
        p.plot(x, y_flow, color='green', linewidth=2)
        p.set_ylabel('flow(mm/h)')
        plt.savefig(os.path.join(definitions.ROOT_DIR, 'example/rain_flow_event_'+str(start_time).split(' ')[0]+'_wy.png'))
    '''
    # XXX_FLOW 和 XXX_RAIN 长度不同，原因暂时未知，可能是数据本身问题（如插值导致）或者单位未修整
    plt.figure()
    x = time
    rain_event_array = np.zeros(shape=len(time))
    flow_event_array = np.zeros(shape=len(time))
    for i in range(0, len(BEGINNING_RAIN)):
        rain_event = filtered_rain_aver_df['rain'][BEGINNING_RAIN[i]: END_RAIN[i]]
        beginning_index = np.argwhere(time == BEGINNING_RAIN[i])[0][0]
        end_index = np.argwhere(time == END_RAIN[i])[0][0]
        rain_event_array[beginning_index: end_index + 1] = rain_event
    for i in range(0, len(BEGINNING_FLOW)):
        flow_event = flow_mm_h[BEGINNING_FLOW[i]: END_FLOW[i]]
        beginning_index = np.argwhere(time == BEGINNING_FLOW[i])[0][0]
        end_index = np.argwhere(time == END_FLOW[i])[0][0]
        flow_event_array[beginning_index: end_index + 1] = flow_event
    y_rain = rain_event_array
    y_flow = flow_event_array
    fig, ax = plt.subplots(figsize=(16, 12))
    p = ax.twinx()  # 包含另一个y轴的坐标轴对象
    ax.bar(x, y_rain, color='red', alpha=0.6)
    ax.set_ylabel('rain(mm)')
    ax.invert_yaxis()
    p.plot(x, y_flow, color='green', linewidth=2)
    p.set_ylabel('flow(mm/h)')
    plt.savefig(os.path.join(definitions.ROOT_DIR, 'example/rain_flow_events.png'))
    '''
    '''
    session_amount = 5
    for i in range(0, session_amount):
        # 雨洪长度可能不一致，姑且长度取最大值
        start = BEGINNING_RAIN[i] if BEGINNING_RAIN[i] < BEGINNING_FLOW[i] else BEGINNING_FLOW[i]
        end = END_RAIN[i] if END_RAIN[i] > END_FLOW[i] else END_FLOW[i]
        rain_session = filtered_rain_aver_array[start:end + 1]
        flow_session_mm_h = flow_mm_h[start:end + 1]
        flow_session_m3_s = flow_m3_s[start:end + 1]
        rain_time = time[start:end + 1]
        session_df = pd.DataFrame(
            {'RAIN_TM': rain_time, 'RAIN_SESSION': rain_session, 'FLOW_SESSION_MM_H': flow_session_mm_h,
             'FLOW_SESSION_M3_S': flow_session_m3_s})
        date_path = np.datetime_as_string(rain_time[0]).split('T')[0] + \
                    np.datetime_as_string(rain_time[0]).split('T')[1].split(':')[0] + \
                    np.datetime_as_string(rain_time[0]).split('T')[1].split(':')[1]
        session_df.to_csv(os.path.join(definitions.ROOT_DIR, 'example/sessions/' +
                                       date_path + '_session.csv'))
        '''


# need fusion with the last test
def test_calibrate_flow():
    # test_biliu_rain_flow_division()
    deap_dir = os.path.join(definitions.ROOT_DIR, 'example/deap_dir/')
    # pet_df含有潜在蒸散发
    pet_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'example/era5_data/pet_calc/PET_result.CSV'), engine='c',
                         parse_dates=['time']).set_index('time')
    session_path = os.path.join(definitions.ROOT_DIR, 'example/sessions/')
    calibrate_df_total = pd.DataFrame()
    for dir_name, sub_dir, files in os.walk(session_path):
        for file in files:
            # session_df 含有雨和洪
            session_df = pd.read_csv(os.path.join(session_path, file), engine='c', parse_dates=['RAIN_TM']).set_index(
                'RAIN_TM').drop(columns=['Unnamed: 0'])
            session_pet = pet_df.loc[session_df.index].to_numpy().flatten()
            calibrate_df = pd.DataFrame({'PRCP': session_df['RAIN_SESSION'].to_numpy(), 'PET': session_pet,
                                         'streamflow': session_df['FLOW_SESSION_M3_S'].to_numpy()})
            calibrate_df_total = pd.concat([calibrate_df_total, calibrate_df], axis=0)
        calibrate_np = calibrate_df_total.to_numpy()
        calibrate_np = np.expand_dims(calibrate_np, axis=0)
        calibrate_np = np.swapaxes(calibrate_np, 0, 1)
        observed_output = np.expand_dims(calibrate_np[:, :, -1], axis=0)
        observed_output = np.swapaxes(observed_output, 0, 1)
        pop = calibrate_by_ga(input_data=calibrate_np[:, :, 0:2], observed_output=observed_output, deap_dir=deap_dir,
                              warmup_length=24)
        print(pop)
        return pop


def plot_rainfall_runoff(
    t,
    p,
    qs,
    fig_size=(8, 6),
    c_lst="rbkgcmy",
    leg_lst=None,
    dash_lines=None,
    title=None,
    xlabel=None,
    ylabel=None,
    linewidth=1,
):
    fig, ax = plt.subplots(figsize=fig_size)
    if dash_lines is not None:
        assert type(dash_lines) == list
    else:
        dash_lines = np.full(len(qs), False).tolist()
    for k in range(len(qs)):
        tt = t[k] if type(t) is list else t
        q = qs[k]
        leg_str = None
        if leg_lst is not None:
            leg_str = leg_lst[k]
        (line_i,) = ax.plot(tt, q, color=c_lst[k], label=leg_str, linewidth=linewidth)
        if dash_lines[k]:
            line_i.set_dashes([2, 2, 10, 2])

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
    # Create second axes, in order to get the bars from the top you can multiply by -1
    ax2 = ax.twinx()
    ax2.bar(tt, -p, color="b")

    # Now need to fix the axis labels
    max_pre = max(p)
    ax2.set_ylim(-max_pre * 5, 0)
    y2_ticks = np.arange(0, max_pre, 20)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels, fontsize=16)
    # ax2.set_yticklabels([lab.get_text()[1:] for lab in ax2.get_yticklabels()])
    if title is not None:
        ax.set_title(title, loc="center", fontdict={"fontsize": 17})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("降水（mm/day）", fontsize=8, loc='top')
    # ax2.set_ylabel("precipitation (mm/day)", fontsize=12, loc='top')
    # https://github.com/matplotlib/matplotlib/issues/12318
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(bbox_to_anchor=(0.01, 0.9), loc="upper left", fontsize=16)
    ax.grid()