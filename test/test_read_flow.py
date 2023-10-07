import os.path

import pandas as pd
import sqlalchemy

import definitions


def test_flow():
    '''
    engine = sqlalchemy.create_engine("mssql+pymssql://jupyterhub_readonly:jupyterhub_readonly@10.55.55.108:1433/rtdb")
    query_stations = "SELECT * FROM rtdb.dbo.ST_PPTN_R WHERE STCD = '21401550'"
    stations_river_stcd = pd.read_sql(query_stations, engine)
    stations_river_stcd.to_csv('biliuriver.csv')
    '''
    engine = sqlalchemy.create_engine(
        "mssql+pymssql://sa:Jsyy123@10.10.50.166:1433/JY2012"
    )
    '''
    query = "select * from STInfo"
    ST_PPTN_STID = pd.read_sql(query, engine)
    ST_PPTN_STID.to_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/biliu_total_stas.csv'), encoding='utf16')
    '''
    query = "select * from Day_Water where STID='4031'"
    data_yushi = pd.read_sql(query, engine)
    data_yushi.to_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/yushi_flow.csv'))