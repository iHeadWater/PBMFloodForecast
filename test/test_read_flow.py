import pandas as pd
import sqlalchemy


def test_flow():
    engine = sqlalchemy.create_engine("mssql+pymssql://jupyterhub_readonly:jupyterhub_readonly@10.55.55.108:1433/rtdb")
    query_stations = "SELECT * FROM rtdb.dbo.ST_PPTN_R WHERE STCD = '21401550'"
    stations_river_stcd = pd.read_sql(query_stations, engine)
    stations_river_stcd.to_csv('biliuriver.csv')