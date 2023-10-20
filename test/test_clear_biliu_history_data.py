import os.path

import pandas as pd

import definitions


def test_clear_biliu_history_data():
    history_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data')
    st_rain_0_df = pd.read_csv(os.path.join(history_path, 'st_rain_c.CSV'), engine='c')
    st_rain_1_df = pd.read_csv(os.path.join(history_path, 'st_rain_c_1.CSV'), names=st_rain_0_df.columns, engine='c')
    st_rain_2_df = pd.read_csv(os.path.join(history_path, 'st_rain_c_2.CSV'), names=st_rain_0_df.columns, engine='c')
    st_rain_3_df = pd.read_csv(os.path.join(history_path, 'st_rain_c_3.CSV'), names=st_rain_0_df.columns, engine='c')
    st_rain_df = pd.concat([st_rain_0_df, st_rain_1_df, st_rain_2_df, st_rain_3_df], axis=0).reset_index().drop(columns=['index',
                                                                                                                         'collecttime'])
    st_water_0_df = pd.read_csv(os.path.join(history_path, 'st_water_c.CSV'), engine='c')
    st_water_1_df = pd.read_csv(os.path.join(history_path, 'st_water_c_1.CSV'), names=st_water_0_df.columns, engine='c')
    st_water_df = pd.concat([st_water_0_df, st_water_1_df], axis=0).reset_index().drop(columns=['index', 'collecttime'])
    stpara_df = pd.read_csv(os.path.join(history_path, 'st_stpara_r.CSV'), engine='c', encoding='gbk')
    splited_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited')
    for para_id in stpara_df['paraid']:
        para_name = (stpara_df['paraname'][stpara_df['paraid'] == para_id]).values[0]
        if para_name != '电压':
            rain_para_df = st_rain_df[st_rain_df['paraid'] == para_id]
            if len(rain_para_df) > 0:
                stid = (stpara_df['stid'][stpara_df['paraid'] == para_id]).values[0]
                rain_para_df.to_csv(os.path.join(splited_path, str(stid)+'_'+str(para_name)+'.csv'))
            water_para_df = st_water_df[st_water_df['paraid'] == para_id]
            if len(water_para_df) > 0:
                stid = (stpara_df['stid'][stpara_df['paraid'] == para_id]).values[0]
                water_para_df.to_csv(os.path.join(splited_path, str(stid)+'_'+str(para_name)+'.csv'))


def test_resample_biliu_data():
    splited_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited')
    splited_hourly_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    for dir_name, sub_dir, files in os.walk(splited_path):
        for file in files:
            csv_path = os.path.join(splited_path, file)
            prcp_df = pd.read_csv(csv_path, engine='c', parse_dates=['systemtime']).set_index(['systemtime']).drop(columns=['Unnamed: 0'])
            if '雨量' in file:
                # resample的时间点是向下取整，例如4：30和4：40的数据会被整合到4：00
                sum_series = prcp_df['paravalue'].resample('H').sum()
                sum_df = pd.DataFrame({'paravalue': sum_series})
                sum_df.to_csv(os.path.join(splited_hourly_path, file.split('.')[0]+'_hourly.csv'))
            if '水位' in file:
                mean_series = prcp_df['paravalue'].resample('H').mean()
                mean_df = pd.DataFrame({'paravalue': mean_series})
                mean_df.to_csv(os.path.join(splited_hourly_path, file.split('.')[0]+'_hourly.csv'))
