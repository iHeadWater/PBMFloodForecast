import os
import pandas as pd
from geopandas import GeoDataFrame

import definitions
import geopandas as gpd
from matplotlib import pyplot as plt


def test_compare_bs_average():
    # 选取金店、桂云花、天益、转山湖、大姜屯
    test_dict = {'4002': '21423132', '4003': '21423100', '4006': '21423000',
                 '4010': '21423050', '4015': '21422600'}
    gdf_biliu_shp: GeoDataFrame = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp'
                                                                                   '/碧流河流域.shp'), engine='pyogrio')
    biliu_series_dict = {}
    sl_series_dict = {}
    start_date = '2022-08-07'
    end_date = '2022-08-25'
    test_range = pd.date_range(start_date, end_date, freq='D')
    test_range_time = pd.date_range('2022-08-07 00:00:00', '2022-08-25 00:00:00', freq='D')
    exam_biliu_data = os.path.join(definitions.ROOT_DIR, 'example/biliu_rain_daily_datas')
    exam_sl_data = os.path.join(definitions.ROOT_DIR, 'example/songliao_exam_stas')
    for key in test_dict.keys():
        biliu_rain = pd.read_csv(os.path.join(exam_biliu_data, key+'_biliu_rain.csv'), engine='c', parse_dates=['InsertTime'])
        sl_rain = pd.read_csv(os.path.join(exam_sl_data, test_dict[key]+'_rain.csv'), engine='c', parse_dates=['TM'])
        biliu_rain['InsertTime'] = biliu_rain['InsertTime'].dt.date
        biliu_rain = biliu_rain.set_index('InsertTime')
        sl_rain = sl_rain.set_index('TM')
        biliu_rain_list = biliu_rain.loc[test_range, 'Rainfall'].to_list()
        sl_time_list = sl_rain.loc[test_range_time, 'DRP'].fillna(0).to_list()
        biliu_series_dict[key] = biliu_rain_list
        sl_series_dict[test_dict[key]] = sl_time_list
    biliu_df = pd.DataFrame(biliu_series_dict)
    sl_df = pd.DataFrame(sl_series_dict)
    voronoi_gdf = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'example/biliuriver_shp/碧流河流域_副本.shp'))
    voronoi_gdf = voronoi_gdf.set_index('STCD')
    stcd_area_dict = {}
    for stcd in test_dict.values():
        polygon = voronoi_gdf.loc[stcd]['geometry']
        area = polygon.area*12100
        stcd_area_dict[stcd] = area
    rain_aver_list_biliu = []
    for i in range(0, len(biliu_df)):
        rain_aver = 0
        for stcd in biliu_df.columns:
            rain_aver += (biliu_df.iloc[i])[stcd] * stcd_area_dict[test_dict[stcd]]/gdf_biliu_shp['area'][0]
        rain_aver_list_biliu.append(rain_aver)
    rain_aver_list_sl = []
    for i in range(0, len(sl_df)):
        rain_aver = 0
        for stcd in sl_df.columns:
            rain_aver += (sl_df.iloc[i])[stcd] * stcd_area_dict[stcd]/gdf_biliu_shp['area'][0]
        rain_aver_list_sl.append(rain_aver)
    result = pd.DataFrame({'Date': test_range, 'Biliu': rain_aver_list_biliu, 'SL': rain_aver_list_sl}, columns=['Date', 'Biliu', 'SL'])
    # result = result.set_index('Date')
    result.plot(marker='o')
    plt.xlabel('Date')
    plt.ylabel('Rainfall')
    plt.show()