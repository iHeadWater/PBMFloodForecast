import os.path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import whitebox
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import Point

import definitions
from hydromodel.division.dmca_esr import STEP1_STEP2_Tr_and_fluctuations_timeseries, STEP3_core_identification, \
    STEP4_end_rain_events, STEP6_checks_on_rain_events, STEP7_end_flow_events, STEP8_beginning_flow_events, \
    STEP5_beginning_rain_events, STEP9_checks_on_flow_events, STEP10_checks_on_overlapping_events
from xaj.calibrate_ga_xaj_bmi import calibrate_by_ga


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


def get_rain_average():
    '''
    engine = sqlalchemy.create_engine("mssql+pymssql://jupyterhub_readonly:jupyterhub_readonly@10.55.55.108:1433/rtdb")
    query_stations = "SELECT STCD,STNM,LGTD,LTTD FROM rtdb.dbo.ST_STBPRP_B WHERE STTP = 'PP'"
    pp_df = pd.read_sql(query_stations, engine)
    '''
    pp_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/rain_stations.csv'), engine='c').drop(
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
    gdf_biliu_shp: GeoDataFrame = gpd.read_file(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp'
                                                                                   '/碧流河流域.shp'), engine='pyogrio')
    gdf_rain_stations = gpd.sjoin(gdf_pps, gdf_biliu_shp, 'inner', 'intersects')
    gdf_rain_stations = gdf_rain_stations[~(gdf_rain_stations['STCD'] == '21422950')]
    gdf_rain_stations.to_file(
        os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/biliu_basin_rain_stas.shp'))
    rain_path = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/rain_datas/')
    '''
    pp_list = gdf_rain_stations['STCD'].tolist()
    for i in pp_list:
        query_stations = "SELECT * FROM rtdb.dbo.ST_PPTN_R WHERE STCD ='{}';".format(i)
        stations_river_stcd = pd.read_sql(query_stations, engine)
        stations_river_stcd.to_csv(os.path.join(rain_path, i+'_rain.csv'))
    '''
    series_dict = {}
    # start_date = '2014-01-01'
    # end_date = '2022-12-31'
    for root, dirs, files in os.walk(rain_path):
        for file in files:
            stcd = file.split('_')[0]
            rain_table = pd.read_csv(os.path.join(rain_path, file), engine='c')
            rain_table['TM'] = pd.to_datetime(rain_table['TM'])
            # 参差不齐，不能按照时间选择，先按照索引硬对
            # drp_series = rain_table['DRP'].loc[(rain_table['TM'] > pd.to_datetime(start_date)) & (rain_table['TM'] < pd.to_datetime(end_date))].fillna(0).tolist()
            # 21422950站点不能用, 要移除
            drp_series = rain_table['DRP'][(rain_table.index >= 5000) & (rain_table.index <= 14130)].fillna(0).tolist()
            series_dict[stcd] = drp_series
    origin_rain_data = pd.DataFrame(series_dict)
    # 读取voronoi文件
    voronoi_gdf = get_voronoi()
    stcd_area_dict = {}
    for i in range(0, len(voronoi_gdf)):
        polygon = voronoi_gdf.geometry[i]
        area = polygon.area * 12100
        stcd_area_dict[voronoi_gdf['STCD'][i]] = area
    rain_aver_list = []
    for i in range(0, len(origin_rain_data)):
        rain_aver = 0
        for stcd in origin_rain_data.columns:
            rain_aver += (origin_rain_data.iloc[i])[stcd] * stcd_area_dict[stcd] / gdf_biliu_shp['area'][0]
        rain_aver_list.append(rain_aver)
    return rain_aver_list


def get_infer_inq():
    '''
    engine = sqlalchemy.create_engine("mssql+pymssql://jupyterhub_readonly:jupyterhub_readonly@10.55.55.108:1433/rtdb")
    query_stations = "SELECT * FROM rtdb.dbo.ST_RSVR_R WHERE STCD = '21401550'"
    stations_river_stcd = pd.read_sql(query_stations, engine)
    stations_river_stcd.to_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliu_rsvr.csv'))
    '''
    biliu_flow_df = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_rsvr.csv'),
                                engine='c', parse_dates=['TM'])
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
    new_df.to_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliu_inq_interpolated.csv'))
    return new_df['INQ'].to_numpy()


def test_biliu_rain_flow_division():
    rain = get_rain_average()
    flow = get_infer_inq()
    rain_min = 0.02
    max_window = 100
    # rain和flow之间的索引要尽量“对齐”
    time = range(0, len(rain))
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = STEP1_STEP2_Tr_and_fluctuations_timeseries(rain, flow,
                                                                                                      rain_min,
                                                                                                      max_window)
    beginning_core, end_core = STEP3_core_identification(fluct_bivariate_Tr)
    end_rain = STEP4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min)
    beginning_rain = STEP5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min)
    beginning_rain_checked, end_rain_checked, beginning_core, end_core = STEP6_checks_on_rain_events(beginning_rain,
                                                                                                     end_rain, rain,
                                                                                                     rain_min,
                                                                                                     beginning_core,
                                                                                                     end_core)
    end_flow = STEP7_end_flow_events(end_rain_checked, beginning_core, end_core, rain, fluct_rain_Tr, fluct_flow_Tr, Tr)
    beginning_flow = STEP8_beginning_flow_events(beginning_rain_checked, end_rain_checked, rain, beginning_core,
                                                 fluct_rain_Tr, fluct_flow_Tr)
    beginning_flow_checked, end_flow_checked = STEP9_checks_on_flow_events(beginning_rain_checked, end_rain_checked,
                                                                           beginning_flow,
                                                                           end_flow, fluct_flow_Tr)
    BEGINNING_RAIN, BEGINNING_FLOW, END_RAIN, END_FLOW = STEP10_checks_on_overlapping_events(beginning_flow_checked,
                                                                                             end_flow_checked,
                                                                                             beginning_flow_checked,
                                                                                             end_flow_checked, time)
    return BEGINNING_RAIN, BEGINNING_FLOW, END_RAIN, END_FLOW


def test_calibrate_flow():
     #calibrate_by_ga()


def get_voronoi():
    origin_basin_shp = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/碧流河流域.shp')
    dup_basin_shp = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/碧流河流域_副本.shp')
    shutil.copyfile(origin_basin_shp, dup_basin_shp)
    node_shp = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/biliu_basin_rain_stas.shp')
    voronoi_from_shp(src=node_shp, des=dup_basin_shp)
    voronoi_gdf = gpd.read_file(dup_basin_shp, engine='pyogrio')
    return voronoi_gdf
