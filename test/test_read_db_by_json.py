import json
import os.path

import sqlalchemy as sqa
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import sessionmaker, scoped_session

import definitions


def test_read_database_json():
    json_path = os.path.join(definitions.ROOT_DIR, 'example/database.json')
    with (open(json_path, 'r') as jsonp):
        json_obj = json.load(jsonp)
        sl_dict = json_obj['biliu_old']
        sl_database_link = sl_dict['class'] + '+' + sl_dict['driver'] + '://' + sl_dict['id'] + ':' + sl_dict[
            'password'] + '@' + sl_dict['ip'] + ':' + sl_dict['port'] + '/' + sl_dict['database']
        sl_engine = sqa.create_engine(sl_database_link)
        '''
        query = "select * from STInfo"
        ST_PPTN_STID = pd.read_sql(query, sl_engine)
        ST_PPTN_STID.to_csv('biliu_total_stas.csv')
        '''
        md = MetaData()
        md.reflect(bind=sl_engine)
        st_table = md.tables['STInfo']
        table_obj = Table(st_table, md, autoload_with=sl_engine)
        session = scoped_session(sessionmaker(bind=sl_engine))
        result = session.query(table_obj).filter_by(STID=4000).all()
        print(result)

