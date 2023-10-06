import os.path
import shutil
from os.path import relpath

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import whitebox
from geopandas import GeoDataFrame
from shapely import Point
import sqlalchemy

import definitions


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


def test_rain_average():
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
        area = polygon.area*12100
        stcd_area_dict[voronoi_gdf['STCD'][i]] = area
    rain_aver_list = []
    for i in range(0, len(origin_rain_data)):
        rain_aver = 0
        for stcd in origin_rain_data.columns:
            rain_aver += (origin_rain_data.iloc[i])[stcd] * stcd_area_dict[stcd]/gdf_biliu_shp['area'][0]
        rain_aver_list.append(rain_aver)
    return rain_aver_list


def get_voronoi():
    origin_basin_shp = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/碧流河流域.shp')
    dup_basin_shp = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/碧流河流域_副本.shp')
    shutil.copyfile(origin_basin_shp, dup_basin_shp)
    node_shp = os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliuriver_shp/biliu_basin_rain_stas.shp')
    voronoi_from_shp(src=node_shp, des=dup_basin_shp)
    voronoi_gdf = gpd.read_file(dup_basin_shp, engine='pyogrio')
    return voronoi_gdf
